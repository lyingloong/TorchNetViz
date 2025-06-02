$model = "DoubleCNN"
$model_file = "DoubleCNN"

$args_json_obj = @{
    type  = "configs"
    value = @{
        task_name           = "long_term_forecast"
        features            = "M"
        seq_len             = 96
        pred_len            = 96
        use_norm            = $true
        enc_in              = 7
        time_kernel_size    = 3
        variable_kernel_size= 3
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
