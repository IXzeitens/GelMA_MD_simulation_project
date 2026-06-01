# Gelatin MD Pipeline

自動化建構 Gelatin-水箱分子動力學模擬的前處理與 NAMD 執行流程：從 PDB 模型開始，依設定濃度計算所需水分子數與正立方體邊長，透過 Packmol 填充、PSFGEN 建立拓樸，最後交由 NAMD2 跑 NVT / NPT。

## 目錄結構

```
\root
 ├── config.json                    # 系統設定與計算結果
 ├── main.py                        # 主流程控制
 ├── \input
 │    └── OOO.pdb                   # 輸入模型，例 Gelatin.pdb / Gel1MA_23.pdb
 ├── \packmol
 │    ├── packmol.exe               # Packmol 可執行檔
 │    ├── packmol_config.inp        # 自動產生
 │    └── water.pdb
 ├── \script                        # Python package + NAMD/力場資產
 │    ├── __init__.py
 │    ├── paths.py                  # ProjectPaths 目錄解析
 │    ├── config.py                 # PipelineConfig dataclass + I/O
 │    ├── pdb_utils.py              # PDB 低階工具（計原子數、取元素）
 │    ├── calculation.py            # 分子量 / 水分子數 / 箱體邊長計算
 │    ├── segments.py               # 切割規劃 + PDB 切檔
 │    ├── packmol_runner.py         # Packmol 輸入檔生成 + 執行
 │    ├── psfgen_runner.py          # 動態 TCL 生成 + VMD 執行
 │    ├── namd_runner.py            # NAMD 設定補丁 + 執行
 │    ├── pipeline.py               # Step 0–6 orchestrator
 │    ├── top_all36_prot_HYP_caf_3.txt   # 拓樸檔（需手動放入）
 │    ├── par_all36*.prm            # 力場參數檔
 │    ├── NVT.conf
 │    ├── npt_1.conf
 │    └── npt_2.conf
 ├── \temp                          # 執行時自動產生
 │    ├── OOO_wb.pdb                # Packmol 輸出
 │    ├── GA.pdb / GB.pdb / ...     # 依 chain 數切割
 │    └── WT3.pdb / WT4.pdb / ...   # 每 10000 個水分子一份
 └── \Output                        # 執行時自動產生
      ├── debug_1.pdb / debug_1.psf # PSFGEN 產生
      ├── NVT.conf / npt_1.conf / npt_2.conf   # 依 L 動態更新
      ├── nvt.log / npt_1.log / npt_2.log
      └── system_nvt.dcd / ...      # NAMD 軌跡
```

## 先決條件

- Python 3.8+
- [Packmol](https://m3g.github.io/packmol/)
- [VMD](https://www.ks.uiuc.edu/Research/vmd/)（提供 `psfgen`）
- [NAMD2](https://www.ks.uiuc.edu/Research/namd/)
- `script/top_all36_prot_HYP_caf_3.txt` 需手動放入

若 `packmol` / `vmd` / `namd2` 不在 PATH，於 `config.json` 的 `software_paths` 指定絕對路徑。

## config.json 欄位

| 欄位 | 說明 |
| --- | --- |
| `chain_count` | Gel-chain 數量（Step 0 互動輸入可覆寫） |
| `concentration` | 目標 wt%（Step 0 互動輸入可覆寫） |
| `polymer_density_g_cm3` | 聚合物密度，預設 1.3 |
| `water_density_g_cm3` | 水密度，預設 0.997 |
| `packmol_expansion_factor` | 箱體膨脹係數，預設 1.05 |
| `software_paths.packmol/vmd/namd` | 可執行檔路徑 |
| `calculated_M` | Step 1 寫回：單條 chain 分子量 (g/mol) |
| `calculated_n_water` | Step 1 寫回：所需水分子數 |
| `calculated_box_L_Angstrom` | Step 1 寫回：正立方體邊長 (Å，偶數) |
| `atoms_per_chain` | Step 1 寫回：單條 chain 原子數 |
| `segments` | Step 4 寫回：各 segment 起訖原子編號 |

## 使用方式

1. 將輸入模型放入 `input/`（單一 `.pdb`）。
2. 確認 `script/top_all36_prot_HYP_caf_3.txt` 存在。
3. 執行：

   ```bash
   python main.py
   ```

4. 依提示輸入 chain 數量與目標濃度（直接 Enter 使用 config.json 現值）。

完成後，模擬輸出位於 `Output/`。

## 流程總覽

| 步驟 | 模組 | 動作 |
| --- | --- | --- |
| 0 | `pipeline.prompt_user_settings` | 詢問使用者 chain 數與濃度，寫入 config.json |
| 1 | `calculation.compute_system_params` | 計算 M、n_water、L，寫回 config.json |
| 2 | `packmol_runner.write_input_file` | 組 `packmol/packmol_config.inp`（箱體 `-L/2 ~ L/2`） |
| 3 | `packmol_runner.run_packmol` | 產生 `temp/OOO_wb.pdb` |
| 4 | `segments.plan_segments` + `split_pdb` | 依 chain 原子數切出 `GA/GB/GC...`；其餘每 10000 個水分子切一份 `WT3/WT4/...` |
| 5 | `psfgen_runner.write_tcl` + `run_vmd` | 動態生成 TCL，VMD psfgen → `Output/debug_1.pdb` + `debug_1.psf` |
| 6 | `namd_runner.prepare_configs` + `run_stage` | 以更新後的 PBC / PMEGrid 執行 `NVT.conf` → `npt_1.conf` → `npt_2.conf` |

### 動態更新的 NAMD 設定

- `cellBasisVector{1,2,3}` 以 Step 1 計算之 `L` 覆寫。
- `PMEGridSize{X,Y,Z}` 取 ≥ L 的最小可被 2/3/5 因式分解整數。
- `parameters` 路徑自動指向 `../script/`。

## 常見問題

- **PDB 讀不到**：確認 `input/` 下只放一個 `.pdb`（目前會取第一個）。
- **Packmol 卡住**：tolerance 預設 2.5 Å、box 已含 1.05 膨脹係數；如擁擠可調整 `packmol_expansion_factor`。
- **WT segment 編號從 3 開始**：沿用原有慣例（保留 WT1 / WT2 給其他可能用途）。
