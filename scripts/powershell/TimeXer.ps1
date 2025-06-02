$model = "TimeXer"
$model_file = "TimeXer"

$init_obj = @{
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

$input_shapes_obj = @{
    x_enc        = @(1, 96, 7)
    x_mark_enc   = @(1, 96, 1)
    x_dec        = @(1, 128, 96)
    x_mark_dec   = @(1, 128, 1)
}

$init_str = $init_obj | ConvertTo-Json -Compress -Depth 3
$input_shapes_str = $input_shapes_obj | ConvertTo-Json -Compress -Depth 3

$encoded_init = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($init_str))
$encoded_input_shapes = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($input_shapes_str))

$arg_string = "[`'$encoded_init`']"

python main.py `
    --model "$model" `
    --model_file "$model_file" `
    --args "$arg_string" `
    --input_dim_sizes "1,7,96,128"`
    --input_shapes "$encoded_input_shapes"
#    *> TimeXer_output.txt
