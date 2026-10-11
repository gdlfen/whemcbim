# 第三轮接口探测（真实网络实测）


## A.东财资金流通道变体
[https无ut] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[http无ut] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[加Referer] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[加ut] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2主机] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2delay主机] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[cb参数] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[fields1全] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[OK] A.东财资金流通道变体


## B.akshare 个股资金流
ak.stock_individual_fund_flow 失败: ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
rank 失败: ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[OK] B.akshare 个股资金流


## C.新浪资金流备选
[ssl_qsfx_zjlrqs] status=200 前 500: [{"opendate":"2026-10-09","trade":"1263.4500","changeratio":"0.00609975","turnover":"27.7781","netamount":"632979412.0300","ratioamount":"0.143711","r0_net":"643814361.2200","r0_ratio":"0.14617106","r0x_ratio":"87.6104","cnt_r0x_ratio":"1","cate_ra":"0.0583949","cate_na":"970718166.1300"},{"opendate":"2026-10-08","trade":"1255.0000","changeratio":"-0.00287617","turnover":"19.7552","netamount":"-81073007.0500","ratioamount":"-0.02627","r0_net":"-93822659.3900","r0_ratio":"-0.03040130","r0x_ratio"
[ssl_bkzj_zjlrqs] status=200 前 500: []
[ssl_qsfx_lscjfb] status=200 前 500: [{"opendate":"2026-10-09","trade":"1263.4500","changeratio":"0.00609975","turnover":"27.7781","netamount":"632979412.0300","ratioamount":"0.143711","r0":"3160151920.9800","r1":"1150801891.3900","r2":"88465222.0000","r3":"0.0000","r0_net":"643814361.2200","r1_net":"-17008023.1900","r2_net":"6173074.0000","r3_net":"0.0000"},{"opendate":"2026-10-08","trade":"1255.0000","changeratio":"-0.00287617","turnover":"19.7552","netamount":"-81073007.0500","ratioamount":"-0.02627","r0":"1987228617.1700","r1":
[OK] C.新浪资金流备选


## D.炸板池/涨停池端点
[getTopicZBPool 20261011] tc=14 pool=14
   字段: ['c', 'm', 'n', 'p', 'ztp', 'zdp', 'amount', 'ltsz', 'tshare', 'hs', 'fbt', 'zbc', 'zf', 'zs', 'zttj', 'hybk']
   样例: {"c": "605303", "m": 1, "n": "园林股份", "p": 32210, "ztp": 34680, "zdp": 2.1566760540008545, "amount": 849336400, "ltsz": 5193456911.68, "tshare": 5193456911.68, "hs": 16.301061630249023, "fbt": 92502, "zbc": 1, "zf": 17.760862350463867, "zs": 0.0, "zttj": {"days": 4, "ct": 3}, "hybk": "基础建设"}
[getTopicFBPool 20261011] status=404 前80=
[getTopicFBPool 20261010] status=404 前80=
[getTopicFBPool 20261009] status=404 前80=
[getTopicFBPool 20261008] status=404 前80=
[getTopicFBPool 20261007] status=404 前80=
[getTopicFBPool 20261006] status=404 前80=
[OK] D.炸板池/涨停池端点


## E.akshare 利润表（营业利润/研发）
stock_profit_sheet_by_report_em 签名: (symbol: str = 'SH600519') -> pandas.DataFrame
stock_financial_benefit_ths 签名: (symbol: str = '000063', indicator: str = '按报告期') -> pandas.DataFrame
stock_profit_sheet_by_yearly_em 签名: (symbol: str = 'SH600519') -> pandas.DataFrame
shape=(103, 203)
关注列: ['REPORT_DATE', 'REPORT_DATE_NAME', 'TOTAL_OPERATE_INCOME', 'TOTAL_OPERATE_INCOME_YOY', 'OPERATE_INCOME', 'OPERATE_INCOME_YOY', 'TOTAL_OPERATE_COST', 'TOTAL_OPERATE_COST_YOY', 'OPERATE_COST', 'OPERATE_COST_YOY', 'RESEARCH_EXPENSE', 'RESEARCH_EXPENSE_YOY', 'ME_RESEARCH_EXPENSE', 'ME_RESEARCH_EXPENSE_YOY', 'OPERATE_PROFIT_OTHER', 'OPERATE_PROFIT_OTHER_YOY', 'OPERATE_PROFIT_BALANCE', 'OPERATE_PROFIT_BALANCE_YOY', 'OPERATE_PROFIT', 'OPERATE_PROFIT_YOY', 'TOTAL_PROFIT_BALANCE', 'TOTAL_PROFIT_BALANCE_YOY', 'TOTAL_PROFIT', 'TOTAL_PROFIT_YOY', 'PARENT_NETPROFIT', 'PARENT_NETPROFIT_YOY', 'DEDUCT_PARENT_NETPROFIT', 'DEDUCT_PARENT_NETPROFIT_YOY']
           REPORT_DATE REPORT_DATE_NAME  TOTAL_OPERATE_INCOME  TOTAL_OPERATE_INCOME_YOY  OPERATE_INCOME  OPERATE_INCOME_YOY  TOTAL_OPERATE_COST  TOTAL_OPERATE_COST_YOY  OPERATE_COST  OPERATE_COST_YOY  RESEARCH_EXPENSE  RESEARCH_EXPENSE_YOY  ME_RESEARCH_EXPENSE  ME_RESEARCH_EXPENSE_YOY  OPERATE_PROFIT_OTHER  OPERATE_PROFIT_OTHER_YOY  OPERATE_PROFIT_BALANCE  OPERATE_PROFIT_BALANCE_YOY  OPERATE_PROFIT  OPERATE_PROFIT_YOY  TOTAL_PROFIT_BALANCE  TOTAL_PROFIT_BALANCE_YOY  TOTAL_PROFIT  TOTAL_PROFIT_YOY  PARENT_NETPROFIT  PARENT_NETPROFIT_YOY  DEDUCT_PARENT_NETPROFIT  DEDUCT_PARENT_NETPROFIT_YOY
0  2026-06-30 00:00:00           2026中报          9.227807e+10                  1.300099    9.070326e+10            1.469869        3.094604e+10                9.141923  9.473763e+09         21.810009      1.149262e+08             55.512605                  NaN                      NaN                   NaN                       NaN                     0.0                         NaN    6.141129e+10           -2.162982                   0.0                       NaN  6.143842e+10         -2.133839      4.451688e+10             -1.951595             4.446421e+10                    -2.040174
1  2026-03-31 00:00:00          2026一季报          5.470291e+10                  6.336009    5.390925e+10            6.538007        1.719222e+10               19.115008  5.520729e+09         35.930656      5.931072e+07            125.085303                  NaN                      NaN                   NaN 
[OK] E.akshare 利润表（营业利润/研发）


## F.VIX via yfinance(tz cache 修复)
已设置 yf tz cache
[^VIX] 失败 YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')
[VIX] 失败 YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')
[OK] F.VIX via yfinance(tz cache 修复)


## G.股吧人气/搜索变体
-93bb-38", "marketType": "", "srcSecurityCode": "600519.SH", "pageNo": 1, "pageSize": 100} -> {"globalId":"786e4c21-70dc-435a-93bb-38","message":"OK","status":0,"code":0,"data":[],"stack":null}
a-93bb-38", "marketType": "", "srcSecurityCode": ["600519"], "pageNo": 1, "pageSize": 100} -> {"globalId":"","message":"服务异常","status":-1,"code":-2,"data":null,"stack":null}
ype": "1", "srcSecurityCode": "600519", "pageNo": 1, "pageSize": 100, "deviceType": "web"} -> {"globalId":"786e4c21-70dc-435a-93bb-38","message":"OK","status":0,"code":0,"data":[],"stack":null}
70dc-435a-93bb-38", "marketType": "", "srcSecurityCode": "", "pageNo": 1, "pageSize": 100} -> {"globalId":"786e4c21-70dc-435a-93bb-38","message":"srcSecurityCode 股票代码不能为空","status":-1,"code":-1,"data":null,"stack":null}
rank 页长度=7648，含 'rank' 关键字位置: -1
片段: var env = '' </script>          <link rel="stylesheet" href="//gbfek.dfcfw.com/deploy/rank_web/work/rank_home.css?v=22" />          <title>东方财富人气排行榜</title>     <base target="_blank" /> </head> <!-- render at 2026-10-11 02:06:30,escaped 0ms -->  <body id="body" class="body">     <div id="header">     <!-- 微信分享img -->     <img id="weixin-share" src="//cmsjs.eastmoney.com/common/weixin-share.png"
search(params 方式) 前 700: jQuery35108723733748578402_1693632913001({
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
[OK] G.股吧人气/搜索变体


## H.腾讯快照扩展字段确认
600519 贵州茅台 价=1263.00 PE=19.39 PB=6.28 流通市值=15788.53 总市值=15788.53 [58]=25.2600 [62]=-6.39 [63]=2.10 [64]=4.12 [65]=32.41 [66]=27.30 [67]=1539.98 [68]=1151.01 [69]=-0.31 [70]=-2.76 [71]=6.84 [74]=81.25 [75]=-9.48 [79]=-8.80
000001 平安银行 价=11.59 PE=5.18 PB=0.49 流通市值=2249.12 总市值=2249.15 [58]=4.2883 [62]=7.30 [63]=2.57 [64]=5.25 [65]=7.93 [66]=0.72 [67]=11.89 [68]=9.74 [69]=2.02 [70]=-0.35 [71]=13.17 [74]=71.21 [75]=5.07 [79]=9.81
601318 中国平安 价=52.83 PE=6.00 PB=0.95 流通市值=5631.71 总市值=9566.27 [58]=110.9430 [62]=-19.55 [63]=-0.79 [64]=5.17 [65]=15.50 [66]=1.28 [67]=72.15 [68]=45.92 [69]=-0.97 [70]=-7.35 [71]=8.91 [74]=83.00 [75]=-25.30 [79]=2.86
[OK] H.腾讯快照扩展字段确认


## I.北向报表全量行
行数=1
字段: ['SECURITY_INNER_CODE', 'SECUCODE', 'TRADE_DATE', 'SECURITY_CODE', 'SECURITY_NAME', 'MUTUAL_TYPE', 'CHANGE_RATE', 'CLOSE_PRICE', 'HOLD_SHARES', 'HOLD_MARKET_CAP', 'A_SHARES_RATIO', 'HOLD_SHARES_RATIO', 'FREE_SHARES_RATIO', 'TOTAL_SHARES_RATIO', 'HOLD_MARKETCAP_CHG1', 'HOLD_MARKETCAP_CHG5', 'HOLD_MARKETCAP_CHG10']
首行: {"SECURITY_INNER_CODE": "1000002162", "SECUCODE": "600519.SH", "TRADE_DATE": "2026-06-30 00:00:00", "SECURITY_CODE": "600519", "SECURITY_NAME": "贵州茅台", "MUTUAL_TYPE": "001", "CHANGE_RATE": -0.792495146281, "CLOSE_PRICE": 1185.49, "HOLD_SHARES": 53711656, "HOLD_MARKET_CAP": 63674631071.44, "A_SHARES_RATIO": 4.3, "HOLD_SHARES_RATIO": 4.29, "FREE_SHARES_RATIO": 4.2967, "TOTAL_SHARES_RATIO": 4.2967, "HOLD_MARKETCAP_CHG1": null, "HOLD_MARKETCAP_CHG5": null, "HOLD_MARKETCAP_CHG10": null}
末行 TRADE_DATE=2026-06-30 00:00:00
[OK] I.北向报表全量行