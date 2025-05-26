$model = "TimeXer"
$model_file = "TimeXer"

# 定义 JSON 对象
$json_obj = @{
    type  = "configs"
    value = @{
        task_name  = "long_term_forecast"
        features   = "M"
        seq_len    = 96
        pred_len   = 96
        use_norm   = $true
        patch_len  = 16
        d_model    = 128
        dropout    = 0.1
        enc_in     = 7
        factor     = 3
        n_heads    = 4
        d_ff       = 256
        e_layers   = 2
        activation = "gelu"
        embed      = "fixed"
        freq       = "h"
    }
}

# 转换为压缩 JSON 字符串（不含换行）
$json_str = $json_obj | ConvertTo-Json -Compress -Depth 3

# Base64 编码
$encoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($json_str))

# 构造 JSON list: ["base64字符串"]
$arg_string = "[`'$encoded`']"

# 执行 Python 脚本，注意 --args 参数用引号括住
python main.py `
    --model "$model" `
    --model_file "$model_file" `
    --args "$arg_string" `
    --input_dim_sizes "1,7,16,96,128,256"
