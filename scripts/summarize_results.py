"""Publish a local, self-contained HTML summary and synchronize the status file."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
import plotly.express as px
import pandas as pd
from phm.data import ROOT, read_json, write_json
from phm.training import list_runs

def main():
    runs=list_runs(); chosen={}
    for run in runs:
        m=read_json(run/'metrics.json')
        if m['task'] not in chosen: chosen[m['task']]=(run,m)
    now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    status={'date':now,'rflymad':{'real_dataset_downloaded':False,'real_validation_completed':False,'note':'Use configured legacy adapter; current UAV benchmark is TII.'}}
    sections=['<h1>无人系统 PHM · 初始原型验证报告</h1>',f'<p>生成时间：{now}。真实公开数据；窗口级指标不等同跨机体工程验证。</p>']
    first_plot=True
    for task,(run,m) in chosen.items():
        sections.append(f'<h2>{html.escape(m["dataset"])}</h2><p>实验：{run.name}</p>')
        if task=='uav':
            score=m['results']['test']; ev=m['results']['anomaly_events']
            status['tii_uav']={'real_dataset_downloaded':True,'real_validation_completed':True,'run_id':run.name,'macro_f1':score['macro_f1'],'accuracy':score['accuracy']}
            sections.append(pd.DataFrame([{'测试 Macro-F1':score['macro_f1'],'准确率':score['accuracy'],'任务总数':m['missions'],'健康测试小时':ev['healthy_exposure_hours'],'误报事件/小时':ev['false_alarm_events_per_hour']}]).to_html(index=False))
            fig=px.imshow(score['confusion_matrix'],text_auto=True,title='测试集混淆矩阵（标签为损坏螺旋桨数量）',labels={'x':'预测','y':'真实'})
        else:
            status['nasa_battery']={'real_dataset_downloaded':True,'real_validation_completed':True,'run_id':run.name,'mae_cycles':m['random_forest']['mae'],'rmse_cycles':m['random_forest']['rmse'],'interval_coverage':m['interval']['empirical_coverage']}
            sections.append(pd.DataFrame(m['per_battery']).T.to_html())
            sections.append(f'<p>汇总 MAE {m["random_forest"]["mae"]:.2f} 循环；RMSE {m["random_forest"]["rmse"]:.2f} 循环；探索性区间实测覆盖率 {m["interval"]["empirical_coverage"]:.1%}。未观察到失效的 B0007 不计入 RUL 评分。</p>')
            p=pd.read_parquet(run/'predictions.parquet')
            fig=px.line(p,x='cycle_idx',y=['true_rul','predicted_rul'],facet_col='battery_id',title='逐电池留一验证 · 真实与预测剩余循环')
        sections.append(fig.to_html(full_html=False,include_plotlyjs=True if first_plot else False)); first_plot=False
        sections.append('<ul>'+''.join(f'<li>{html.escape(x)}</li>' for x in m['limitations'])+'</ul>')
        sections.append('<details><summary>完整指标与验证口径</summary><pre>'+html.escape(json.dumps(m,ensure_ascii=False,indent=2))+'</pre></details>')
    synthetic=read_json(ROOT/'source/artifacts/metrics.json')['classification']['macro avg']['f1-score']
    status['synthetic_starter']={'synthetic_only':True,'reported_macro_f1':synthetic,'note':'Historical file, not rerun and not displayed as real-data score.'}
    write_json(ROOT/'validation_status.json',status)
    target=ROOT/'runs/validation_report.html'
    target.write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>PHM 验证报告</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172a45}h1,h2{color:#123d5a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dce4ef;padding:10px}pre{white-space:pre-wrap}li{margin:8px 0}</style><body>'+''.join(sections)+'</body></html>',encoding='utf-8')
    print(target)
if __name__=='__main__':main()
