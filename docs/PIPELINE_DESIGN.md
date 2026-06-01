Data structure
---
\root
 ├── config.json
 ├── main.py
 ├── \input
 │    └── OOO.pdb (e.g. Gelatin.pdb/ Gel1MA_23.pdb ... )
 ├── \packmol
 │    ├── packmol.exe
 │    ├── packmol_config.inp (自動產生)
 │    └── water.pdb
 ├── \script
 │    ├── calculation.py
 │    ├── top_all36_prot_HYP_caf_3.txt (PSFGEN所需拓樸檔，請手動放入)
 │    ├── Packmol_process.py
 │    ├── packmol_psfgen.tcl
 │    ├── NVT.conf
 │    ├── npt_1.conf
 │    └── npt_2.conf
 │    
 ├──\temp (執行時自動產生)
 │    ├── GA/GB/GC/WT3.pdb (Packmol_process.py 生成檔案)
 │    ├── 
 │    └── 
 ├──\Output
 │    ├── debug_1.pdb (Packmol_psfgen.tcl 生成)
 │    ├── debug_1.psf (Packmol_psfgen.tcl 生成)
 │    ├── nvt.dcd ...
 │    ├── npt_1.dcd ...
 │    └── npt_2.dcd ...

Environmet:
NAMD version: NAMD_3.0.2_Win64-multicore-CUDA


Workflow 1
---
(\root\Main.py)
0. 要求使用者設定 Gel-chain 數量 以及目標濃度，寫入 \root\config.json
1. 調用 \script\calculation.py
    1-a. 讀取 \root\input\ OOO.pdb (eg. Gelatin.pdb/ Gel1MA_23.pdb ...)
    1-b. 讀取 \root\ 下的 config.json，取得設定濃度 與 OOO.pdb 的設定數量，以及單條 Gel 的總原子數
    1-c. 計算並顯示出 
        (i)     原子量 M
        (ii)    達到對應濃度所需水分子數 n
        (iii)   使整個水箱落於 1g/cm^3 所需要的正立方體邊長 L (整數)
    1-d. 將前述資訊寫出到 config.json 
2. 將 \root\config.json 的 L 與 n 輸出到 \root\packmol\packmol.inp ，並將輸入檔案 改為讀取到的 OOO.pdb
    2-a. 正立方體水箱上下界以 (-L/2 ~ L/2) 表示
    2-b. 所需調用的 water.pdb 已經放置 \root\packmol\ 下
3. 調用 \root\packmol\packmol.exe，在 \root\Output\ 生成對應的 OOO_wb.pdb
4. 調用 \root\script\packmol_process.py
    4-a. 讀取 config.json 中的單條 chain 總原子數
    4-b. 依據該原子數，切割出設定的 chain 數(GA,GB,GC...)
    4-c. 剩餘水分子數，則每 10000 個水分子切割成一份 WT3 (0~9999-WT3, 10000~19999-WT4...etc)
5. 以 VMD 指令調用 \root\script\packmol_psfgen.tcl
    5-a. 自動依據步驟4 所切割的檔案數生成對應指令列
    5-b. 啟動 psfgen，將生成檔案 debug_1.pdb 與 debug_1.psf 放置於 \root\output\ 下
6. 以 NAMD3 指令調用 \root\script\nvt.conf
    6-a. 啟動 namd2 指令前，需先將步驟 2 中所使用的邊界大小導入 NVT.conf, NPT_1.conf, NPT_2.conf 三者中

Workflow 2 
---
若已經存在有對應的 debug_1.pdb 與 debug_1.psf 時，使用者可以調用根目錄下的 NPT_conti.py 或 auto_extend.py 來接續模擬
(A). (/root/NPT_conti.py)
    1. 調用在 main.py 中已經生成好的 debug_1.psf 與 debug_1.pdb 以及 system_nvt.dcd 系列檔案
    2. 檢查與啟動 npt_1.conf 與 npt_2.conf 進行模擬

(B). (/root/auto_extend.py)
    1. 調用與檢查已經完成的 npt_n (n = 任意整數)系列檔案，參考 npt_2.conf 生成對應的 npt_n+1.conf (n >= 3)
    2. 確保此 auto_extend.py 可以調用 /root/下的 Gelatin、Gel1MA、Gel2MA、Gel3MA 中的資料接續進行計算(參考 main.py)
    3. 啟動 npt_n.conf 進行模擬輸出 system_npt_partn.dcd 系列檔案