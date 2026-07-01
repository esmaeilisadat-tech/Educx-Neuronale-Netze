$dir = "C:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_05_Backpropagation\notebooks"

function Add-ImageOutput {
    param($json, $cellIndex, $imagePath)
    if (Test-Path $imagePath) {
        $bytes = [System.IO.File]::ReadAllBytes($imagePath)
        $base64 = [System.Convert]::ToBase64String($bytes)
        
        $dataObj = New-Object PSObject
        $dataObj | Add-Member -MemberType NoteProperty -Name "text/plain" -Value @("<Figure size 1400x500 with 2 Axes>")
        $dataObj | Add-Member -MemberType NoteProperty -Name "image/png" -Value $base64
        
        $output = New-Object PSObject
        $output | Add-Member -MemberType NoteProperty -Name "output_type" -Value "display_data"
        $output | Add-Member -MemberType NoteProperty -Name "metadata" -Value @{}
        $output | Add-Member -MemberType NoteProperty -Name "data" -Value $dataObj
        
        $json.cells[$cellIndex].outputs = @($output)
    }
}

# Anfaenger
$path = "$dir\Anfaenger.ipynb"
$json = Get-Content -Raw -Path $path | ConvertFrom-Json
Add-ImageOutput $json 3 "$dir\lernrate_einfluss.png"
Add-ImageOutput $json 5 "$dir\optimierer_vergleich.png"
$json | ConvertTo-Json -Depth 100 | Set-Content -Path $path

# Fortgeschrittene
$path = "$dir\Fortgeschrittene.ipynb"
$json = Get-Content -Raw -Path $path | ConvertFrom-Json
Add-ImageOutput $json 1 "$dir\adam_vs_sgd.png"
Add-ImageOutput $json 3 "$dir\lr_scheduler.png"
Add-ImageOutput $json 5 "$dir\gradient_clipping.png"
$json | ConvertTo-Json -Depth 100 | Set-Content -Path $path

# Experte
$path = "$dir\Experte.ipynb"
$json = Get-Content -Raw -Path $path | ConvertFrom-Json
Add-ImageOutput $json 1 "$dir\second_order.png"
Add-ImageOutput $json 3 "$dir\sattelPunkte.png"
Add-ImageOutput $json 5 "$dir\lr_finder.png"
$json | ConvertTo-Json -Depth 100 | Set-Content -Path $path
