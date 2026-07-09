# FINAL CHAIN: wait for 2025-26 windows -> extraction(+dates) -> attach 2025 ->
# v3 (8 seasons) -> THE FINAL BACKTEST. Then 2016/17 enrichment attach.
Set-Location D:\optbot
$log = "artifacts\final_chain.log"
"[$(Get-Date)] waiting for 2025-26 build marker..." | Out-File $log -Encoding utf8
while (-not (Select-String -Path "C:\Users\lilli\Downloads\API\build_2025_windows_v1.log" -Pattern "V1BUILD_DONE" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
"[$(Get-Date)] 2025 build DONE. re-extraction (8 seasons, +dates)" | Out-File $log -Append
Remove-Item artifacts\people_outcomes_*.parquet -Force -ErrorAction SilentlyContinue
python scripts\02c_extract_people_and_outcomes.py 20182019 20192020 20202021 20212022 20222023 20232024 20242025 20252026 *>> $log
"[$(Get-Date)] v2 rejoin" | Out-File $log -Append
python scripts\02d_join_perfect_v2.py *>> $log
"[$(Get-Date)] MP attach 2025" | Out-File $log -Append
python scripts\14_mp_attach_shots.py --year 2025 --write *>> $log
"[$(Get-Date)] v3 assemble (8 seasons)" | Out-File $log -Append
python scripts\16_assemble_v3.py --seasons 20182019 20192020 20202021 20212022 20222023 20232024 20242025 20252026 *>> $log
"[$(Get-Date)] FINAL BACKTEST" | Out-File $log -Append
python scripts\18_final_backtest.py *>> $log
"[$(Get-Date)] CHAIN_COMPLETE" | Out-File $log -Append
