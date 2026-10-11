# 接口探测结果（真实网络实测）


## 1.腾讯实时快照 qt.gtimg.cn
RAW(前 400 字): v_sh600519="1~贵州茅台~600519~1263.00~1255.79~1255.42~35111~19932~15179~1263.00~73~1262.99~2~1262.98~3~1262.88~1~1262.83~8~1263.06~1~1263.11~3~1263.43~1~1263.44~1~1263.45~3~~20261009161459~7.21~0.57~1282.00~1255.42~1263.00/35111/4453223189~35111~445322~0.28~19.39~~1282.00~1255.42~2.12~15788.53~15788.53~6.28~1381.37~1130.21~1.18~78~1268.34~17.73~19.18~~~0.07~445322.3189~25.2600~2~   A~GP-A~-6.39~2.10~4
字段数: 88
  [0] = 1
  [1] = 贵州茅台
  [2] = 600519
  [3] = 1263.00
  [4] = 1255.79
  [5] = 1255.42
  [6] = 35111
  [7] = 19932
  [8] = 15179
  [9] = 1263.00
  [10] = 73
  [11] = 1262.99
  [12] = 2
  [13] = 1262.98
  [14] = 3
  [15] = 1262.88
  [16] = 1
  [17] = 1262.83
  [18] = 8
  [19] = 1263.06
  [20] = 1
  [21] = 1263.11
  [22] = 3
  [23] = 1263.43
  [24] = 1
  [25] = 1263.44
  [26] = 1
  [27] = 1263.45
  [28] = 3
  [29] = 
  [30] = 20261009161459
  [31] = 7.21
  [32] = 0.57
  [33] = 1282.00
  [34] = 1255.42
  [35] = 1263.00/35111/4453223189
  [36] = 35111
  [37] = 445322
  [38] = 0.28
  [39] = 19.39
  [40] = 
  [41] = 1282.00
  [42] = 1255.42
  [43] = 2.12
  [44] = 15788.53
  [45] = 15788.53
  [46] = 6.28
  [47] = 1381.37
  [48] = 1130.21
  [49] = 1.18
  [50] = 78
  [51] = 1268.34
  [52] = 17.73
  [53] = 19.18
  [54] = 
  [55] = 
  [56] = 0.07
  [57] = 445322.3189
  [58] = 25.2600
  [59] = 2
  [60] =    A
  [61] = GP-A
  [62] = -6.39
  [63] = 2.10
  [64] = 4.12
  [65] = 32.41
  [66] = 27.30
  [67] = 1539.98
  [68] = 1151.01
  [69] = -0.31
  [70] = -2.76
  [71] = 6.84
  [72] = 1250081601
  [73] = 1250081601
  [74] = 81.25
  [75] = -9.48
  [76] = 1250081601
  [77] = 
  [78] = 
  [79] = -8.80
  [80] = -0.02
  [81] = 
  [82] = CNY
  [83] = 0
  [84] = ___D__F__N
  [85] = 1263.59
  [86] = -21
  [87] = 
[OK] 1.腾讯实时快照 qt.gtimg.cn


## 2.东财日频资金流 fflow/daykline
[FAIL] 2.东财日频资金流 fflow/daykline
rlopen(
        method=request.method,
    ...<9 lines>...
        chunked=chunked,
    )
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\connectionpool.py", line 841, in urlopen
    retries = retries.increment(
        method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
    )
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\util\retry.py", line 490, in increment
    raise reraise(type(error), error, _stacktrace)
          ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\util\util.py", line 38, in reraise
    raise value.with_traceback(tb)
urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 40, in probe
    fn()
    ~~^^
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 76, in t_fflow_day
    d = js(url)
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 53, in js
    r = get(url, **kw)
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 49, in get
    return S.get(url, **kw)
           ~~~~~^^^^^^^^^^^
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))



## 3.东财分时资金流 fflow/kline klt=1
[FAIL] 3.东财分时资金流 fflow/kline klt=1
rlopen(
        method=request.method,
    ...<9 lines>...
        chunked=chunked,
    )
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\connectionpool.py", line 841, in urlopen
    retries = retries.increment(
        method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
    )
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\util\retry.py", line 490, in increment
    raise reraise(type(error), error, _stacktrace)
          ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\33564\AppData\Local\Programs\Python\Python313\Lib\site-packages\urllib3\util\util.py", line 38, in reraise
    raise value.with_traceback(tb)
urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 40, in probe
    fn()
    ~~^^
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 97, in t_fflow_min
    d = js(url)
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 53, in js
    r = get(url, **kw)
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 49, in get
    return S.get(url, **kw)
           ~~~~~^^^^^^^^^^^
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))



## 4.东财所属板块 slist(spt=3)
顶层: {"rc": 0, "rt": 18, "svr": 183120645, "lt": 1, "full": 1, "dlmkts": "", "dsc": "0", "data": {"total": 28, "diff": {"0": {"f3": 204, "f12": "BK0665", "f13": 90, "f14": "电商概念", "f62": 2493146880.0, "f184": 284}, "1": {"f3": 131, "f12": "BK1637", "f13": 90, "f14": "东方财富热股", "f62": 2396741632.0, "f184": 104}, "2": {"f3": 174, "f12": "BK0834", "f13": 90, "f14": "乡村振兴", "f62": 2069702352.0, "f184": 536}, "3": {"f3": 86, "f12": "BK0999", "f13": 90, "f14": "茅指数", "f62": 1283035904.0, "f184": 136}, "4": {"f3": 98, "f12": "BK0590", "f13": 90, "f14": "西部大开发", "f62": 909409792.0, "f184": 45}, "5": {"f3": 95, "f12": "BK0811", "f13": 90, "f14": "超级品牌", "f62": 849409728.0, "f184": 333}, "6": {"f3": 155, "f12": "BK1653", "f13": 90, "f14": "味蕾经济", "f62": 287063632.0, "f184": 99}, "7": {"f3": 82, "f12": "BK
板块数: 28
[FAIL] 4.东财所属板块 slist(spt=3)
Traceback (most recent call last):
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 40, in probe
    fn()
    ~~^^
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 116, in t_slist
    for it in diff[:12]:
              ~~~~^^^^^
KeyError: slice(None, 12, None)



## 4b.东财所属板块 slist(仅代码)
{"rc": 102, "rt": 1, "svr": 183119994, "lt": 1, "full": 1, "dlmkts": "", "dsc": "0", "data": null}
[OK] 4b.东财所属板块 slist(仅代码)


## 5.东财板块成分 clist(b:BK0475)
只数: 10
  {"f3": 0.47, "f9": 6.97, "f12": "001227", "f14": "兰州银行", "f20": 12074877996, "f23": 0.39}
  {"f3": 0.32, "f9": 6.2, "f12": "002966", "f14": "苏州银行", "f20": 42158342764, "f23": 0.8}
  {"f3": 0.0, "f9": 4.69, "f12": "002936", "f14": "郑州银行", "f20": 15729318049, "f23": 0.34}
  {"f3": -0.16, "f9": 5.35, "f12": "600015", "f14": "华夏银行", "f20": 100264049348, "f23": 0.31}
  {"f3": -0.18, "f9": 4.36, "f12": "601997", "f14": "贵阳银行", "f20": 20584395168, "f23": 0.31}
  {"f3": -0.34, "f9": 5.11, "f12": "601916", "f14": "浙商银行", "f20": 79922090652, "f23": 0.44}
  {"f3": -0.39, "f9": 5.01, "f12": "601860", "f14": "紫金银行", "f20": 9445649035, "f23": 0.45}
  {"f3": -0.56, "f9": 5.32, "f12": "600928", "f14": "西安银行", "f20": 15777777780, "f23": 0.43}
  {"f3": -0.57, "f9": 8.34, "f12": "601288", "f14": "农业银行", "f20": 2442881576434, "f23": 0.85}
  {"f3": -0.58, "f9": 5.59, "f12": "601528", "f14": "瑞丰银行", "f20": 10046266342, "f23": 0.5}
[OK] 5.东财板块成分 clist(b:BK0475)


## 6.东财 RPT_LICO_FN_CPD 财务
success=True message=ok
行数: 3
字段: ['SECURITY_CODE', 'SECURITY_NAME_ABBR', 'TRADE_MARKET_CODE', 'TRADE_MARKET', 'SECURITY_TYPE_CODE', 'SECURITY_TYPE', 'UPDATE_DATE', 'REPORTDATE', 'BASIC_EPS', 'DEDUCT_BASIC_EPS', 'TOTAL_OPERATE_INCOME', 'PARENT_NETPROFIT', 'WEIGHTAVG_ROE', 'YSTZ', 'SJLTZ', 'BPS', 'MGJYXJJE', 'XSMLL', 'YSHZ', 'SJLHZ', 'ASSIGNDSCRPT', 'PAYYEAR', 'PUBLISHNAME', 'ZXGXL', 'NOTICE_DATE', 'ORG_CODE', 'TRADE_MARKET_ZJG', 'ISNEW', 'QDATE', 'DATATYPE', 'DATAYEAR', 'DATEMMDD', 'EITIME', 'SECUCODE', 'BOARD_NAME', 'ORI_BOARD_CODE', 'BOARD_CODE']
{"SECURITY_CODE": "600519", "SECURITY_NAME_ABBR": "贵州茅台", "TRADE_MARKET_CODE": "069001001001", "TRADE_MARKET": "上交所主板", "SECURITY_TYPE_CODE": "058001001", "SECURITY_TYPE": "A股", "UPDATE_DATE": "2026-08-15 00:00:00", "REPORTDATE": "2026-06-30 00:00:00", "BASIC_EPS": 35.57, "DEDUCT_BASIC_EPS": 35.53, "TOTAL_OPERATE_INCOME": 92278072083.21, "PARENT_NETPROFIT": 44516880421.86, "WEIGHTAVG_ROE": 16.75, "YSTZ": 1.3000994756, "SJLTZ": -1.95, "BPS": 200.989754763617, "MGJYXJJE": 56.54890853726, "XSMLL": 89.5552128279, "YSHZ": -31.3105, "SJLHZ": -36.5904, "ASSIGNDSCRPT": "不分配不转增", "PAYYEAR": "2026", "PUBLISHNAME": "白酒Ⅱ", "ZXGXL": null, "NOTICE_DATE": "2026-08-15 00:00:00", "ORG_CODE": "10002602", "TRADE_MARKET_ZJG": "0101", "ISNEW": "1", "QDATE": "2026Q2", "DATATYPE": "2026年 半年报", "DATAYEAR": "2026", "DATEMMDD": "半年报", "EITIME": "2026-08-14 20:42:07", "SECUCODE": "600519.SH", "BOARD_NAME": "白酒Ⅱ", "ORI_BOARD_CODE": "1277", "BOARD_CODE": "BK1277"}
[OK] 6.东财 RPT_LICO_FN_CPD 财务


## 7.akshare stock_financial_abstract
shape=(80, 105)
列前 12: ['选项', '指标', '20260630', '20260331', '20251231', '20250930', '20250630', '20250331', '20241231', '20240930', '20240630', '20240331']
列后 6: ['20020331', '20011231', '20010630', '20001231', '19991231', '19981231']
     选项           指标      20260630      20260331      20251231      20250930      20250630      20250331      20241231      20240930      20240630      20240331      20231231      20230930      20230630      20230331      20221231      20220930      20220630      20220331      20211231      20210930      20210630      20210331      20201231      20200930      20200630      20200331      20191231      20190930      20190630      20190331      20181231      20180930      20180630      20180331      20171231      20170930      20170630      20170331      20161231      20160930      20160630      20160331      20151231      20150930      20150630      20150331      20141231      20140930      20140630      20140331      20131231      20130930      20130630      20130331      20121231      20120930      20120630      20120331      20111231      20110930      20110630      20110331      20101231      20100930      20100630      20100331      20091231      20090930      20090630      20090331      20081231      20080930      20080630      20080331      20071231      20070930      20070630      20070331      20061231      20060930      20060630      20060331      20051231      20050930      20050630      20050331      20041231      20040930      20040630      20040331      20031231      20030930      20030630      20030331      20021231      20020930      20020630      20020331      20011231      20010630      20001231      19991231      19981231
0  常用指标        归母净利润  4.451688e+10  2.724251e+10  8.232007e+10  6.462675e+10  4.540296e+10  2.684747e+10  8.622815e+10  6.082755e+10  4.169561e+10  2.406526e+10  7.473407e+10  5.287622e+10  3.598042e+10  2.079488e+10  6.271747e+10  4.439982e+10  2.979391e+10  1.724497e+10  5.246014e+10  3.726617e+10  2.465399e+10  1.395446e+10  4.669729e+10  3.382710e+10  2.260166e+10  1.309377e+10  4.120647e+10  3.045486e+10  1.995103e+10  1.122143e+10  3.520363e+10  2.473355e+10  1.576419e+10  8.506907e+09  2.707936e+10  1.998385e+10  1.125086e+10  6.123119e+09  1.671836e+10  1.246558e+10  8.802637e+09  4.889272e+09  1.550309e+10  1.142464e+10  7.888232e+09  4.364902e+09  1.534980e+10  1.069333e+10  7.229854e+09  3.699386e+09  1.513664e+10  1.107019e+10  7.247945e+09  3.593098e+09  1.330808e+10  1.042006e+10  6.995722e+09  2.969216e+09  8.763146e+09  6.569085e+09  4.907177e+09  1.883995e+09  5.051194e+09  4.174174e+09  3.099757e+09  1.265663e+09  4.312446e+09  3.786951e+09  2.790276e+09  1.217009e+09  3.799481e+09  3.144759e+09  2.23956
指标去重前 40: ['归母净利润', '营业总收入', '营业成本', '净利润', '扣非净利润', '股东权益合计(净资产)', '商誉', '经营现金流量净额', '基本每股收益', '每股净资产', '每股现金流', '净资产收益率(ROE)', '总资产报酬率(ROA)', '毛利率', '销售净利率', '期间费用率', '资产负债率', '稀释每股收益', '摊薄每股收益_最新股数', '摊薄每股净资产_期末股数', '调整每股净资产_期末股数', '每股净资产_最新股数', '每股经营现金流', '每股现金流量净额', '每股企业自由现金流量', '每股股东自由现金流量', '每股未分配利润', '每股资本公积金', '每股盈余公积金', '每股留存收益', '每股营业收入', '每股营业总收入', '每股息税前利润', '摊薄净资产收益率', '净资产收益率_平均', '净资产收益率_平均_扣除非经常损益', '摊薄净资产收益率_扣除非经常损益', '息税前利润率', '总资产报酬率', '总资本回报率']
[OK] 7.akshare stock_financial_abstract


## 8a.akshare stock_zh_a_gdhs_detail_em
shape=(63, 15)
列: ['股东户数统计截止日', '区间涨跌幅', '股东户数-本次', '股东户数-上次', '股东户数-增减', '股东户数-增减比例', '户均持股市值', '户均持股数量', '总市值', '总股本', '股本变动', '股本变动原因', '股东户数公告日期', '代码', '名称']
    股东户数统计截止日      区间涨跌幅  股东户数-本次  股东户数-上次  股东户数-增减  股东户数-增减比例        户均持股市值        户均持股数量           总市值         总股本  股本变动 股本变动原因    股东户数公告日期      代码    名称
0  2013-03-22 -19.122572    69331    49870    19461  39.023461  2.531398e+06  14974.253941  1.755043e+11  1038180000     0    NaN  2013-03-29  600519  贵州茅台
1  2013-03-31  -0.112393    68539    69331     -792  -1.142346  2.557771e+06  15147.288405  1.753071e+11  1038180000     0    NaN  2013-04-18  600519  贵州茅台
2  2013-06-30  17.668854    53045    68539   -15494 -22.606107  3.765005e+06  19571.684419  1.997147e+11  1038180000     0    NaN  2013-08-31  600519  贵州茅台
[OK] 8a.akshare stock_zh_a_gdhs_detail_em


## 8b.东财 RPT_HOLDERNUMLATEST
行数: 1
字段: ['SECURITY_CODE', 'SECUCODE', 'SECURITY_NAME_ABBR', 'HOLDER_NUM', 'PRE_HOLDER_NUM', 'HOLDER_NUM_CHANGE', 'HOLDER_NUM_RATIO', 'INTERVAL_CHRATE', 'END_DATE', 'PRE_END_DATE', 'PRE_E_DATE', 'AVG_MARKET_CAP', 'AVG_HOLD_NUM', 'TOTAL_MARKET_CAP', 'TOTAL_A_SHARES', 'HOLD_NOTICE_DATE', 'HOLD_N_DATE', 'CHANGE_SHARES', 'CHANGE_REASON', 'CLOSE_PRICE', 'TRADE_MARKET_CODE', 'ORG_CODE', 'SECURITY_TYPE_CODE']
{"SECURITY_CODE": "600519", "SECUCODE": "600519.SH", "SECURITY_NAME_ABBR": "贵州茅台", "HOLDER_NUM": 296404, "PRE_HOLDER_NUM": 243159, "HOLDER_NUM_CHANGE": 53245, "HOLDER_NUM_RATIO": 21.897194839591, "INTERVAL_CHRATE": -16.30735947, "END_DATE": "2026-06-30 00:00:00", "PRE_END_DATE": "2026-03-31 00:00:00", "PRE_E_DATE": "03/31", "AVG_MARKET_CAP": 4999794.99996454, "AVG_HOLD_NUM": 4217.49234490762, "TOTAL_MARKET_CAP": 1481959237169.49, "TOTAL_A_SHARES": 1250081601, "HOLD_NOTICE_DATE": "2026-08-15 00:00:00", "HOLD_N_DATE": "08/15", "CHANGE_SHARES": -2188614, "CHANGE_REASON": "资产重组", "CLOSE_PRICE": 1185.49, "TRADE_MARKET_CODE": "069001001001", "ORG_CODE": "10002602", "SECURITY_TYPE_CODE": "058001001"}
[OK] 8b.东财 RPT_HOLDERNUMLATEST


## 9.东财 RPTA_WEB_RZRQ_GGMX 两融
行数: 3
字段: ['DATE', 'MARKET', 'SCODE', 'SECNAME', 'RZYE', 'RQYL', 'RZRQYE', 'RQYE', 'RQMCL', 'RZRQYECZ', 'RZMRE', 'SZ', 'RZYEZB', 'RZMRE3D', 'RZMRE5D', 'RZMRE10D', 'RZCHE', 'RZCHE3D', 'RZCHE5D', 'RZCHE10D', 'RZJME', 'RZJME3D', 'RZJME5D', 'RZJME10D', 'RQMCL3D', 'RQMCL5D', 'RQMCL10D', 'RQCHL', 'RQCHL3D', 'RQCHL5D', 'RQCHL10D', 'RQJMG', 'RQJMG3D', 'RQJMG5D', 'RQJMG10D', 'SPJ', 'ZDF', 'RCHANGE3DCP', 'RCHANGE5DCP', 'RCHANGE10DCP', 'KCB', 'TRADE_MARKET_CODE', 'TRADE_MARKET', 'FIN_BALANCE_GR', 'SECUCODE']
{"DATE": "2026-10-09 00:00:00", "MARKET": "融资融券_沪证", "SCODE": "600519", "SECNAME": "贵州茅台", "RZYE": 17086652613, "RQYL": 204448, "RZRQYE": 17344870437, "RQYE": 258217824, "RQMCL": 1800, "RZRQYECZ": 16828434789, "RZMRE": 250134331, "SZ": 1578853062063, "RZYEZB": 1.0822193, "RZMRE3D": 746466516, "RZMRE5D": 1252837913, "RZMRE10D": 2545451793, "RZCHE": 393467744, "RZCHE3D": 978431825, "RZCHE5D": 1392407181, "RZCHE10D": 2518535110, "RZJME": -143333413, "RZJME3D": -231965309, "RZJME5D": -139569268, "RZJME10D": 26916683, "RQMCL3D": 5800, "RQMCL5D": 13600, "RQMCL10D": 89500, "RQCHL": 2800, "RQCHL3D": 14200, "RQCHL5D": 18200, "RQCHL10D": 30700, "RQJMG": -1000, "RQJMG3D": -8400, "RQJMG5D": -4600, "RQJMG10D": 58800, "SPJ": 1263, "ZDF": 0.5741, "RCHANGE3DCP": 2.2192, "RCHANGE5DCP": 2.1019, "RCHANGE10DCP": -0.3141, "KCB": 0, "TRADE_MARKET_CODE": "069001001001", "TRADE_MARKET": "上交所主板", "FIN_BALANCE_GR": -0.831883506998, "SECUCODE": "600519.SH"}
[OK] 9.东财 RPTA_WEB_RZRQ_GGMX 两融


## 10a.东财 RPT_DAILYBILLBOARD_DETAILSNEW
行数: 2
字段: ['TRADE_DATE', 'DEAL_AMOUNT_RATIO', 'BILLBOARD_DEAL_AMT', 'FREE_MARKET_CAP', 'EXPLAIN', 'SECUCODE', 'SECURITY_CODE', 'CLOSE_PRICE', 'CHANGE_RATE', 'TURNOVERRATE', 'D1_CLOSE_ADJCHRATE', 'D2_CLOSE_ADJCHRATE', 'D5_CLOSE_ADJCHRATE', 'D10_CLOSE_ADJCHRATE', 'SECURITY_NAME_ABBR', 'EXPLANATION', 'BILLBOARD_SELL_AMT', 'BILLBOARD_BUY_AMT', 'SECURITY_INNER_CODE', 'BILLBOARD_NET_AMT', 'DEAL_NET_RATIO', 'ACCUM_AMOUNT', 'MARKET', 'CHANGE_TYPE', 'TRADE_ID', 'BUY_SEAT', 'SELL_SEAT', 'BUY_RATIO', 'SELL_RATIO', 'TRADE_MARKET', 'D30_CLOSE_ADJCHRATE', 'D20_CLOSE_ADJCHRATE', 'TRADE_MARKET_CODE', 'BUY_SEAT_NEW', 'SELL_SEAT_NEW', 'SECURITY_TYPE_CODE', 'SUM_BUY_AMT', 'SUM_SELL_AMT', 'NET_BS_AMT']
{"TRADE_DATE": "2013-01-28 00:00:00", "DEAL_AMOUNT_RATIO": 34.551787584793, "BILLBOARD_DEAL_AMT": 987176704.46, "FREE_MARKET_CAP": 185315130000, "EXPLAIN": "5家机构卖出，成功率48.38%", "SECUCODE": "600519.SH", "SECURITY_CODE": "600519", "CLOSE_PRICE": 178.5, "CHANGE_RATE": -5.581, "TURNOVERRATE": 1.5247, "D1_CLOSE_ADJCHRATE": -0.19047619, "D2_CLOSE_ADJCHRATE": 0.90756303, "D5_CLOSE_ADJCHRATE": -1.88795518, "D10_CLOSE_ADJCHRATE": 0.66666667, "SECURITY_NAME_ABBR": "贵州茅台", "EXPLANATION": "有价格涨跌幅限制的日收盘价格跌幅偏离值达到7%的前三只证券", "BILLBOARD_SELL_AMT": 683034128.39, "BILLBOARD_BUY_AMT": 304142576.07, "SECURITY_INNER_CODE": "1000002162", "BILLBOARD_NET_AMT": -378891552.32, "DEAL_NET_RATIO": -13.261435743254, "ACCUM_AMOUNT": 2857093000, "MARKET": "SH", "CHANGE_TYPE": "137001002002001", "TRADE_ID": 57957, "BUY_SEAT": 13311, "SELL_SEAT": 33333, "BUY_RATIO": 10.64517592077, "SELL_RATIO": 23.906611664024, "TRADE_MARKET": "上交所主板", "D30_CLOSE_ADJCHRATE": -5.07002801, "D20_CLOSE_ADJCHRATE": -2.11764706, "TRADE_MARKET_CODE": "069001001001", "BUY_SEAT_NEW": "13311", "SELL_SEAT_NEW": "33333", "SECURITY_TYPE_CODE": "058001001", "SUM_BUY_AMT": 304142576.07, "SUM_SELL_AMT": 683034128.39, "NET_BS_AMT": -378891552.32}
[OK] 10a.东财 RPT_DAILYBILLBOARD_DETAILSNEW


## 10b.东财 RPT_ORGANIZATION_TRADE_DETAILS
行数: 2
字段: ['SECUCODE', 'SECURITY_NAME_ABBR', 'SECURITY_CODE', 'TRADE_DATE', 'CLOSE_PRICE', 'CHANGE_RATE', 'BUY_TIMES', 'SELL_TIMES', 'BUY_AMT', 'SELL_AMT', 'NET_BUY_AMT', 'ACCUM_AMOUNT', 'RATIO', 'TURNOVERRATE', 'FREECAP', 'EXPLANATION', 'D1_CLOSE_ADJCHRATE', 'D2_CLOSE_ADJCHRATE', 'D3_CLOSE_ADJCHRATE', 'D5_CLOSE_ADJCHRATE', 'D10_CLOSE_ADJCHRATE', 'MARKET', 'TRADE_MARKET_CODE', 'BUY_COUNT', 'SELL_COUNT']
{"SECUCODE": "600519.SH", "SECURITY_NAME_ABBR": "贵州茅台", "SECURITY_CODE": "600519", "TRADE_DATE": "2013-01-28 00:00:00", "CLOSE_PRICE": 178.5, "CHANGE_RATE": -5.581, "BUY_TIMES": 2, "SELL_TIMES": 5, "BUY_AMT": 129177951.52, "SELL_AMT": 683034128.39, "NET_BUY_AMT": -553856176.87, "ACCUM_AMOUNT": 2857093000, "RATIO": -19.385304464013, "TURNOVERRATE": 1.5247, "FREECAP": 1853.15, "EXPLANATION": "有价格涨跌幅限制的日收盘价格跌幅偏离值达到7%的前三只证券", "D1_CLOSE_ADJCHRATE": -0.19047619, "D2_CLOSE_ADJCHRATE": 0.90756303, "D3_CLOSE_ADJCHRATE": 0.22408964, "D5_CLOSE_ADJCHRATE": -1.88795518, "D10_CLOSE_ADJCHRATE": 0.66666667, "MARKET": "沪市", "TRADE_MARKET_CODE": "069001001001", "BUY_COUNT": 2, "SELL_COUNT": 5}
[OK] 10b.东财 RPT_ORGANIZATION_TRADE_DETAILS


## 11a.东财 RPT_MUTUAL_HOLDSTOCKNORTH_STA
success=False msg=HOLD_DATE排序列不存在
行数: 0
[OK] 11a.东财 RPT_MUTUAL_HOLDSTOCKNORTH_STA


## 11b.akshare stock_hsgt_individual_em
[FAIL] 11b.akshare stock_hsgt_individual_em
Traceback (most recent call last):
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 40, in probe
    fn()
    ~~^^
  File "E:\deepseek desktop\个股预测程序\_probe\probe_net.py", line 238, in t_ak_hsgt
    df = ak.stock_hsgt_individual_em(stock="600519")
TypeError: stock_hsgt_individual_em() got an unexpected keyword argument 'stock'



## 12.东财 RPT_F10_EH_FREEHOLDERS
行数: 12
字段: ['SECUCODE', 'SECURITY_CODE', 'ORG_CODE', 'END_DATE', 'HOLDER_NAME', 'HOLD_NUM', 'FREE_HOLDNUM_RATIO', 'HOLD_NUM_CHANGE', 'CHANGE_RATIO', 'IS_HOLDORG', 'HOLDER_RANK', 'SECURITY_NAME_ABBR', 'HOLDER_CODE', 'SECURITY_TYPE_CODE', 'HOLDER_STATE', 'HOLDER_MARKET_CAP', 'HOLD_RATIO', 'HOLD_CHANGE', 'HOLD_RATIO_CHANGE', 'HOLDER_TYPE', 'SHARES_TYPE', 'UPDATE_DATE', 'REPORT_DATE_NAME', 'HOLDER_NEW', 'FREE_RATIO_QOQ', 'HOLDER_STATEE', 'IS_REPORT', 'HOLDER_CODE_OLD', 'HOLDER_NEWTYPE', 'HOLDNUM_CHANGE_NAME', 'IS_MAX_REPORTDATE', 'COOPERATION_HOLDER_MARK', 'MXID', 'LISTING_STATE', 'XZCHANGE', 'NEW_CHANGE_RATIO', 'HOLDER_STATE_NEW', 'HOLD_ORG_CODE_SOURCE', 'HOLD_CHANGE_RATIOTB', 'HOLD_NUM_ABBR', 'IS_DJG', 'IS_SJKZR', 'CAMRELATION_GROUP_LABEL', 'IS_LANDSTOCK', 'HOLDER_RELATION_LABLE', 'LISTED_SHARES_RATIO', 'NOTICE_DATE']
{"SECUCODE": "600519.SH", "SECURITY_CODE": "600519", "ORG_CODE": "10002602", "END_DATE": "2026-06-30 00:00:00", "HOLDER_NAME": "中国贵州茅台酒厂(集团)有限责任公司", "HOLD_NUM": 681282935, "FREE_HOLDNUM_RATIO": 54.499077056651, "HOLD_NUM_CHANGE": "不变", "CHANGE_RATIO": null, "IS_HOLDORG": "1", "HOLDER_RANK": 1, "SECURITY_NAME_ABBR": "贵州茅台", "HOLDER_CODE": "10016801", "SECURITY_TYPE_CODE": "058001001", "HOLDER_STATE": null, "HOLDER_MARKET_CAP": 807654106613.15, "HOLD_RATIO": 54.4991, "HOLD_CHANGE": "不变", "HOLD_RATIO_CHANGE": 0.0953, "HOLDER_TYPE": "其它", "SHARES_TYPE": "A股", "UPDATE_DATE": "2026-08-15 00:00:00", "REPORT_DATE_NAME": "2026中报", "HOLDER_NEW": "10016801", "FREE_RATIO_QOQ": "0.17507769", "HOLDER_STATEE": null, "IS_REPORT": "1", "HOLDER_CODE_OLD": "80010298", "HOLDER_NEWTYPE": "其他", "HOLDNUM_CHANGE_NAME": "不变", "IS_MAX_REPORTDATE": "1", "COOPERATION_HOLDER_MARK": "10016801", "MXID": "b43389a2a668ef37b90bad7787b89da8", "LISTING_STATE": "0", "XZCHANGE": 0, "NEW_CHANGE_RATIO": "不变", "HOLDER_STATE_NEW": "不变", "HOLD_ORG_CODE_SOURCE": "bc808b2d-d2e3-4141-8c29-a80117970840", "HOLD_CHANGE_RATIOTB": 0.4303, "HOLD_NUM_ABBR": "中国贵州茅台酒厂(集团)有限责任公司", "IS_DJG": "0", "IS_SJKZR": "0", "CAMRELATION_GROUP_LABE
第二行: {"SECUCODE": "600519.SH", "SECURITY_CODE": "600519", "ORG_CODE": "10002602", "END_DATE": "2026-06-30 00:00:00", "HOLDER_NAME": "贵州金融控股集团有限责任公司(贵州贵民投资集团有限责任公司)", "HOLD_NUM": 3487220, "FREE_HOLDNUM_RATIO": 0.278959389308, "HOLD_NUM_CHANGE": "新进", "CHANGE_RATIO": null, "IS_HOLDORG": "1", "HOLDER_RANK": 10, "SECURITY_NAME_ABBR": "贵州茅台", "HOLDER_CODE": "10719499", "SECURITY_TYPE_CODE": "058001001", "HOLDER_STATE": "新进", "HOLDER_MARKET_CAP": 4134064437.8, "HOLD_RATIO": 0.279, "HOLD_CHANGE": "新进", "HOLD_RATIO_CHANGE": null, "HOLDER_TYPE": "投资公司", "SHARES_TYPE": "A股", "UPDATE_DATE": "2026-08-15 00:00:00", "REPORT_DATE_NAME": "2026中报", "HOLDER_NEW": "10719499", "FREE_RATIO_QOQ": "新进", "HOLDER_STATEE": "新进", "IS_REPORT": "1", "HOLDER_CODE_OLD": "80684852", "HOLDER_NEWTYPE": "其他", "HOLDNUM_CHANGE_NAM
[OK] 12.东财 RPT_F10_EH_FREEHOLDERS


## 13a.getTopicZTPool
日期 20261011 涨停只数=69
字段: ['c', 'm', 'n', 'p', 'zdp', 'amount', 'ltsz', 'tshare', 'hs', 'lbc', 'fbt', 'lbt', 'fund', 'zbc', 'hybk', 'zttj']
{"c": "002058", "m": 0, "n": "紫竹高科", "p": 24440, "zdp": 9.990999221801758, "amount": 100791831, "ltsz": 3503860934.0800004, "tshare": 3505877234.0800004, "hs": 2.8765933513641357, "lbc": 4, "fbt": 92500, "lbt": 92500, "fund": 89302489, "zbc": 0, "hybk": "电池", "zttj": {"days": 4, "ct": 4}}
[OK] 13a.getTopicZTPool


## 13b.getTopicFBPool
[OK] 13b.getTopicFBPool


## 14.东财 ulist.np 指数
{"rc": 0, "rt": 11, "svr": 183119891, "lt": 1, "full": 1, "dlmkts": "", "dsc": "0", "data": {"total": 3, "diff": [{"f2": 3813.79, "f3": 0.05, "f4": 1.89, "f12": "000001", "f14": "上证指数"}, {"f2": 12641.86, "f3": 0.17, "f4": 20.96, "f12": "399001", "f14": "深证成指"}, {"f2": 3043.33, "f3": 0.22, "f4": 6.67, "f12": "399006", "f14": "创业板指"}]}}
[OK] 14.东财 ulist.np 指数


## 15a.股吧人气排名(100)
status=200
{"globalId":"786e4c21-70dc-435a-93bb-38","message":"srcSecurityCode 股票代码不能为空","status":-1,"code":-1,"data":null,"stack":null}
[OK] 15a.股吧人气排名(100)


## 15b.股吧人气排名(全量)
顶层: {"globalId": "786e4c21-70dc-435a-93bb-38", "message": "srcSecurityCode 股票代码不能为空", "status": -1, "code": -1, "data": null, "stack": null}
条数: 0
[OK] 15b.股吧人气排名(全量)


## 16.东财搜索 jsonp
RAW 前 1200: jQuery35108723733748578402_1693632913001({
    "bizCode": "",
    "bizMsg": "",
    "code": 0,
    "extra": {},
    "hitsTotal": 1,
    "msg": "OK",
    "result": {
        "passportWeb": [{
            "uid": "8223456637392854",
            "stockFollowerCount": 7289,
            "alias": "<em>达利食品</em>",
            "postCount": 0,
            "fansCount": 0,
            "portrait": "https://avator.eastmoney.com/qface/8223456637392854/180",
            "url": "http://iguba.eastmoney.com/8223456637392854",
            "introduction": " ",
            "userFollowingCount": 0,
            "userSelectStockCount": 0
        }]
    },
    "searchId": "b93a1e15-8497-4b4c-a4c8-744305b6f2d8"
  })
[OK] 16.东财搜索 jsonp


## 17.新浪行情 hq.sinajs.cn
[fx_susdcny] -> var hq_str_fx_susdcny="03:00:01,6.6925000000,6.6929000000,6.6929000000,90.0000000000,6.7000000000,6.7002000000,6.6912000000,6.6929000000,在岸人民币,0.0000,0.0000,0.0090,此行情由新浪财经计算得出,0.0000,0.0000,,2026-10-10";
[hf_CL,hf_CAD,hf_GC] -> var hq_str_hf_CL="91.716,,91.640,91.670,92.170,90.010,04:59:58,91.490,91.290,0,3,1,2026-10-10,纽约原油,0";
var hq_str_hf_CAD="14552.970,,14543.000,14551.500,14583.000,14294.500,01:59:59,14540.000,14308.000,0,1,1,2026-10-10,伦铜,17207";
var hq_str_hf_GC="4221.774,,4220.000,4220.400,4233.700,4156.100,04:59:59,4157.000,4159.700,0,5,3,2026-10-10,纽约黄金,0";
[gb_$vix] -> var hq_str_gb_$vix="";
[sh000001] -> var hq_str_sh000001="上证指数,3804.0896,3811.9043,3813.7913,3824.7091,3754.1353,0,0,535149921,888561198264,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-10-09,15:35:24,00,";
[OK] 17.新浪行情 hq.sinajs.cn


## 18a.东财公告 np-anotice
success=1
条数: 5
字段: ['art_code', 'codes', 'columns', 'display_time', 'eiTime', 'language', 'listing_state', 'notice_date', 'product_code', 'sort_date', 'source_type', 'title', 'title_ch', 'title_en']
{"art_code": "AN202608141827994407", "codes": [{"ann_type": "A,SHA", "inner_code": "46077278941243", "market_code": "1", "short_name": "贵州茅台", "stock_code": "600519"}], "columns": [{"column_code": "001002008", "column_name": "其他"}], "display_time": "2026-08-14 20:41:29:380", "eiTime": "2026-08-14 20:42:30:000", "language": "0", "listing_state": "0", "notice_date": "2026-08-15 00:00:00", "product_code": "", "sort_date": "2026-08-15 12:00:00", "source_type": "31", "title": "贵州茅台:贵州茅台关于召开2026年半年度业绩说明会的公告", "title_ch": "贵州茅台:贵州茅台关于召开2026年半年度业绩说明会的公告", "title_en": ""}
[OK] 18a.东财公告 np-anotice


## 19.akshare stock_zh_index_daily
shape=(8738, 6) 列=['date', 'open', 'high', 'low', 'close', 'volume']
            date      open      high       low     close       volume
8735  2026-09-30  3839.253  3851.217  3833.086  3842.195  41456024700
8736  2026-10-08  3838.977  3864.118  3795.374  3811.904  48220168500
8737  2026-10-09  3804.090  3824.709  3754.135  3813.791  53514992100
[OK] 19.akshare stock_zh_index_daily


## 20.akshare 宏观
[cpi_monthly] shape=(357, 5) 列=['商品', '日期', '今值', '预测值', '前值']
            商品          日期   今值  预测值   前值
355  中国CPI月率报告  2025-08-09  0.4  0.3 -0.1
356  中国CPI月率报告  2025-09-10  NaN  NaN  0.4
[ppi_monthly] shape=(248, 4) 列=['月份', '当月', '当月同比增长', '累计']
            月份     当月  当月同比增长      累计
246  2006年02月份  103.0    3.01  103.00
247  2006年01月份  103.1    3.05  103.05
[pmi] shape=(225, 5) 列=['月份', '制造业-指数', '制造业-同比增长', '非制造业-指数', '非制造业-同比增长']
            月份  制造业-指数  制造业-同比增长  非制造业-指数  非制造业-同比增长
223  2008年02月份    53.4  0.564972     59.3  -2.145215
224  2008年01月份    53.0 -3.811252     60.2  -0.331126
[gdp] shape=(82, 9) 列=['季度', '国内生产总值-绝对值', '国内生产总值-同比增长', '第一产业-绝对值', '第一产业-同比增长', '第二产业-绝对值', '第二产业-同比增长', '第三产业-绝对值', '第三产业-同比增长']
             季度  国内生产总值-绝对值  国内生产总值-同比增长  第一产业-绝对值  第一产业-同比增长  第二产业-绝对值  第二产业-同比增长  第三产业-绝对值  第三产业-同比增长
80  2006年第1-2季度    101247.0         13.1    7762.7        4.8   46992.9       14.2   46491.4       13.4
81    2006年第1季度     47831.7         12.4    3012.7        4.4   21418.2       13.1   23400.7       13.0
[lpr] shape=(1576, 5) 列=['TRADE_DATE', 'LPR1Y', 'LPR5Y', 'RATE_1', 'RATE_2']
      TRADE_DATE  LPR1Y  LPR5Y  RATE_1  RATE_2
1574  2026-08-20    3.0    3.5    4.35     4.9
1575  2026-09-20    3.0    3.5    4.35     4.9
[money_supply] shape=(224, 10) 列=['月份', '货币和准货币(M2)-数量(亿元)', '货币和准货币(M2)-同比增长', '货币和准货币(M2)-环比增长', '货币(M1)-数量(亿元)', '货币(M1)-同比增长', '货币(M1)-环比增长', '流通中的现金(M0)-数量(亿元)', '流通中的现金(M0)-同比增长', '流通中的现金(M0)-环比增长']
            月份  货币和准货币(M2)-数量(亿元)  货币和准货币(M2)-同比增长  货币和准货币(M2)-环比增长  货币(M1)-数量(亿元)  货币(M1)-同比增长  货币(M1)-环比增长  流通中的现金(M0)-数量(亿元)  流通中的现金(M0)-同比增长  流通中的现金(M0)-环比增长
222  2008年02月份          421037.84            17.39         0.763839      150177.88        18.95    -3.031337           32454.47             5.96       -11.503457
223  2008年01月份          417846.17            18.88         3.580769      154872.59        20.54     1.543032           36673.15            31.21        20.896562
[OK] 20.akshare 宏观


## 20b.akshare rate_interbank SHIBOR
shape=(4995, 3) 列=['报告日', '利率', '涨跌']
             报告日    利率   涨跌
4992  2026-10-08  1.43  0.0
4993  2026-10-09  1.43  0.0
4994  2026-10-10  1.43  0.0
[OK] 20b.akshare rate_interbank SHIBOR


## 21.yfinance ^VIX
shape=(0, 6)
Empty DataFrame
Columns: [(Adj Close, ^VIX), (Close, ^VIX), (High, ^VIX), (Low, ^VIX), (Open, ^VIX), (Volume, ^VIX)]
Index: []
[OK] 21.yfinance ^VIX