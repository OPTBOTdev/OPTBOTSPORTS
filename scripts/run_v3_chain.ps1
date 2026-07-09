# V3 chain: re-extract (entry+multistint cols) -> v2 join -> v3 assemble + battery
Set-Location D:\optbot
$log = "artifacts\v3_chain.log"
"[$(Get-Date)] re-extraction (02c v2 cols)" | Out-File $log -Encoding utf8
Remove-Item artifacts\people_outcomes_*.parquet -Force -ErrorAction SilentlyContinue
python scripts\02c_extract_people_and_outcomes.py *>> $log
"[$(Get-Date)] v2 rejoin (02d)" | Out-File $log -Append
python scripts\02d_join_perfect_v2.py *>> $log
"[$(Get-Date)] v3 assemble + audit battery (16)" | Out-File $log -Append
python scripts\16_assemble_v3.py *>> $log
"[$(Get-Date)] V3CHAIN_DONE" | Out-File $log -Append
