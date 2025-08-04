# Python ObsPy 教學概覽

本文整理了使用 ObsPy 處理地震資料的核心方法，涵蓋安裝方式、基本資料結構與讀取、繪圖、濾波、降採樣、合併波形、以及從資料中心下載資料等主題。內容來源整理自官方文件與教學，並提供範例程式碼以便快速上手。

## ObsPy 簡介

ObsPy 是一套開源的 Python 函式庫，專門用來讀取、分析與處理地震波形和相關元資料。它支援多種地震資料格式（如 SAC、MiniSEED、GSE2、SEISAN 等），並將檔案匯入成 `Stream` 物件【905473440140812†L72-L85】。每個 `Stream` 物件包含多個 `Trace`（連續波形）物件，每個 `Trace` 又擁有兩個主要屬性：

- **`data` 屬性**：指向 NumPy 陣列，存放實際的時間序列資料【905473440140812†L78-L80】。
- **`stats` 屬性**：保存關於波形的頭資料，例如測站名稱、開始時間、取樣率等【905473440140812†L103-L116】。

透過這些結構，研究人員可以方便地存取、分析和繪製波形資料，也能與地震資料中心進行互動。

## 安裝與版本

截至 2025 年 5 月，ObsPy 的最新穩定版本為 1.4.2，開發者文檔最後更新於 2025‑05‑03【905473440140812†L152-L153】。可以使用 pip 或 conda 安裝：

```bash
pip install obspy
```

或

```bash
conda install -c conda-forge obspy
```

安裝完成後即可在 Python 中匯入 `obspy` 模組並開始使用。

## 基本資料結構與讀取波形

### 讀取檔案

ObsPy 提供 `read()` 函式用來讀取各種地震資料格式並返回 `Stream` 物件【905473440140812†L72-L86】。下面範例讀取一個 GSE2 檔案並檢視其內容：

```python
from obspy import read

# 讀取示例檔案，返回一個 Stream 物件
st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
print(st)  # 列印 Stream 的摘要

# 取得第一個 Trace 物件
tr = st[0]
print(tr)  # 列印 Trace 的摘要

# 讀取頭資料 (stats)
print(tr.stats)  # 包含網路、測站、通道、開始時間、取樣率等

# 取得測站名稱與資料格式
print(tr.stats.station)
print(tr.stats._format)

# 存取實際的波形資料 (NumPy array)
data_array = tr.data
print(len(data_array), data_array[:10])  # 查看樣本數與前十筆資料

# 使用 Stream.plot() 快速預覽波形 (需要 obspy.imaging)
st.plot()
```

此範例展示如何將檔案匯入 `Stream`，並透過 `Trace.stats` 存取頭資料及 `Trace.data` 存取波形陣列【905473440140812†L103-L133】。最後呼叫 `st.plot()` 可以快速預覽波形【905473440140812†L139-L143】。

### `UTCDateTime` 時間物件

ObsPy 的 `UTCDateTime` 類別用來表示精確的 UTC 時間（例如 `tr.stats.starttime`）。此類別支援加減運算以及格式化輸出，適合處理事件時間與資料檢索範圍。

## 波形繪圖

`Stream` 物件內建 `plot()` 方法可直接產生波形圖【905473440140812†L139-L143】。若需要自訂繪圖，可利用 `Trace.times()` 取得時間軸搭配 matplotlib 繪製：

```python
import matplotlib.pyplot as plt
t = tr.times(reftime=tr.stats.starttime)  # 以開始時間為參考
plt.plot(t, tr.data, label="Raw")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Raw seismogram')
plt.legend()
plt.show()
```

## 濾波處理

`Trace` 或 `Stream` 物件的 `filter()` 方法支援 bandpass、bandstop、lowpass 與 highpass 等濾波器【106390039209047†L69-L84】。以下範例採用低通濾波器 (角頻率 1 Hz) 並比較濾波前後的波形：

```python
import numpy as np
import matplotlib.pyplot as plt
import obspy

st = obspy.read('https://examples.obspy.org/RJOB_061005_072159.ehz.new')
tr = st[0]

# 將 Trace 複製一份以保留原始資料
tr_filt = tr.copy()

# 套用低通濾波 (角頻率 1 Hz，兩個角點，zerophase=True)
tr_filt.filter('lowpass', freq=1.0, corners=2, zerophase=True)

# 生成時間軸
t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)

# 繪製原始與濾波後的資料
plt.subplot(2, 1, 1)
plt.plot(t, tr.data, 'k')
plt.ylabel('Raw Data')

plt.subplot(2, 1, 2)
plt.plot(t, tr_filt.data, 'k')
plt.ylabel('Low‑passed Data')
plt.xlabel('Time [s]')
plt.suptitle(str(tr.stats.starttime))
plt.show()
```

## 降採樣 (Decimation)

ObsPy 的 `Trace.decimate()` 方法可將波形以整數因子降採樣。預設會先套用低通濾波避免混疊；如需關閉自動濾波，可使用 `no_filter=True`【96069930715942†L67-L76】。以下範例將 200 Hz 資料降至 50 Hz，並比較不同處理方式：

```python
import numpy as np
import matplotlib.pyplot as plt
import obspy

st = obspy.read('https://examples.obspy.org/RJOB_061005_072159.ehz.new')
tr = st[0]

# 降採樣 (每 4 個點取 1 個)，自動含低通濾波
tr_dec = tr.copy()
tr_dec.decimate(factor=4, strict_length=False)

# 對照只套用相同低通濾波，但不降採樣
tr_filt = tr.copy()
tr_filt.filter('lowpass', freq=0.4 * tr.stats.sampling_rate / 4.0)

# 時間軸
 t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
t_dec = np.arange(0, tr_dec.stats.npts / tr_dec.stats.sampling_rate, tr_dec.stats.delta)

plt.plot(t, tr.data, 'k', label='Raw', alpha=0.3)
plt.plot(t, tr_filt.data, 'b', label='Low‑passed', alpha=0.7)
plt.plot(t_dec, tr_dec.data, 'r', label='Low‑passed/Decimated', alpha=0.7)
plt.xlabel('Time [s]')
plt.xlim(82, 83.5)
plt.title(str(tr.stats.starttime))
plt.legend()
plt.show()
```

## 合併波形

當地震資料被分割成多個檔案時，可使用 `Stream.merge()` 合併為一個連續的波形【724583027429314†L67-L104】。下面示範讀取三個 MiniSEED 片段並合併：

```python
import matplotlib.pyplot as plt
import obspy

# 讀取三個片段，測站 BHE
st = obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE')
st += obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE.1')
st += obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE.2')

# 按開始時間排序
st.sort(['starttime'])

# 以第一個片段的開始時間為參考
 t0 = st[0].stats.starttime
fig, axes = plt.subplots(nrows=len(st)+1, sharex=True)
for tr, ax in zip(st, axes[:-1]):
    ax.plot(tr.times(reftime=t0), tr.data)

# 合併三個片段
st.merge(method=1)
axes[-1].plot(st[0].times(reftime=t0), st[0].data, 'r')
axes[-1].set_xlabel(f'seconds relative to {t0}')
plt.show()
```

`merge()` 方法會按照指定策略（例如優先使用最長片段）將重疊段合併為一條連續波形【724583027429314†L69-L102】。

## 從資料中心下載波形

ObsPy 的 `obspy.clients.fdsn` 模組提供通用的 FDSN 網路服務介面，可從各地震資料中心下載波形、台站與事件資訊【762218662218010†L74-L120】。常見資料類型包括波形資料（MiniSEED 等）、台站資訊（StationXML）與事件資訊（QuakeML）【762218662218010†L114-L119】。如果不確定哪個資料中心包含所需資料，可利用 IRIS Federator 或 EIDAWS Routing Service 等路由服務【762218662218010†L122-L130】。

以下示例展示如何使用 FDSN 客戶端下載特定時段與台站的波形，並套用儀器校正：

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# 建立 FDSN 客戶端 (例如 IRIS)
client = Client('IRIS')

# 定義檢索範圍與台站資訊
starttime = UTCDateTime('2020-01-01T00:00:00')
endtime   = UTCDateTime('2020-01-01T00:02:00')
network   = 'IU'
station   = 'ANMO'
location  = '00'
channel   = 'BH*'  # 三分量

# 下載波形資料與台站元資料
st = client.get_waveforms(network, station, location, channel, starttime, endtime)
inventory = client.get_stations(network=network, station=station,
                                starttime=starttime, endtime=endtime,
                                level='response')

# 移除儀器響應，將計數值轉換為真實單位
st.remove_response(inventory=inventory)
st.plot()
```

在上述範例中，使用 `Client.get_waveforms()` 下載波形，並用 `Client.get_stations()` 取得台站 XML 元資料，再透過 `remove_response()` 移除儀器響應，得到以物理單位表示的波形【762218662218010†L84-L120】。

## 其他功能概述

除了基本功能，ObsPy 還提供多項進階工具：

- **觸發與震相拾取**：`obspy.signal.trigger` 模組支援 STA/LTA、AIC 等演算法，可自動判斷震相到時。
- **波束形成與 FK 分析**：用於陣列資料分析，可估計波源方向與相速度。
- **事件與目錄資料結構**：`obspy.core.event` 模組支援 QuakeML 格式，可建立、修改或輸出地震事件資訊。
- **資料格式轉換**：支援將波形轉換為 ASCII、MATLAB 格式或寫出 MiniSEED 等。

## 結語

本文介紹了 ObsPy 的核心概念與常用操作，包含讀取資料、訪問元資料、繪圖、濾波、降採樣、合併及從資料中心下載波形等功能。透過簡潔的程式碼範例，可以快速掌握 ObsPy 的使用流程，為後續地震資料處理奠定基礎。
