$model = "DoubleCNN"
$model_file = "DoubleCNN"

$args_json_obj = @{
    type  = "configs"
    value = @{
        task_name           = "long_term_forecast"
        features            = "M"                # 多变量输入
        seq_len             = 96                 # 输入序列长度
        pred_len            = 96                 # 输出预测长度
        use_norm            = $true              # 是否使用 RevIN 归一化
        enc_in              = 7                  # 输入特征维度
        time_kernel_size    = 3                  # 时间维度 CNN 的卷积核大小
        variable_kernel_size= 3                  # 变量维度 CNN 的卷积核大小
    }
}

$args_json_str = $args_json_obj | ConvertTo-Json -Compress -Depth 3
$encoded_args = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($args_json_str))

$args_list = "[`'$encoded_args`']"

python main.py `
    --model "$model" `
    --model_file "$model_file" `
    --args "$args_list"`
    --input_dim_sizes "1,7,96"
#    *> DoubleCNN_output.txt
