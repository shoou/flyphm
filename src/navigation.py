"""Sidebar menu routes. Stable URLs preserve the selected page on refresh."""
import streamlit as st

MENUS = {
    '监测中心': [('overview','装备态势'),('alerts','告警与处置')],
    '装备管理': [('assets','装备档案'),('asset_new','新增装备'),('structure','分系统与部件'),('bindings','数据与模型绑定')],
    '健康分析': [('health','装备健康分析'),('replay','故障诊断回放'),('battery','电池健康与寿命'),('imports','数据导入与推理')],
    '维修管理': [('orders','工单与履历'),('incidents','异常事件与复核'),('reviews','历史窗口复核')],
    '模型中心': [('models','模型目录与绑定'),('training','模型训练与验证'),('comparison','模型版本对比')],
}
ROUTES = {route:(group,label) for group,items in MENUS.items() for route,label in items}
ICONS = {'监测中心':':material/dashboard:','装备管理':':material/devices:','健康分析':':material/monitor_heart:','维修管理':':material/build:','模型中心':':material/model_training:'}


def go(route):
    st.query_params['page']=route
    # Replay must not continue running when navigating away from its page.
    if route!='replay':st.session_state['playing']=False


def sidebar_menu():
    route=st.query_params.get('page','overview')
    if route not in ROUTES:route='overview'
    group,label=ROUTES[route]
    with st.container(key='business_menu'):
        for name,items in MENUS.items():
            active=name==group
            st.button(name,icon=ICONS[name],key='menu_'+name,width='stretch',
                      type='primary' if active else 'tertiary',on_click=go,args=(items[0][0],))
            if active:
                with st.container(key='submenu'):
                    for target,title in items:
                        st.button(title,key='nav_'+target,width='stretch',
                                  type='secondary' if route==target else 'tertiary',
                                  icon=':material/chevron_right:' if route==target else None,
                                  on_click=go,args=(target,))
    return route,group,label
