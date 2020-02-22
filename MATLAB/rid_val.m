function out_arr = rid_val(in_arr)
    out_arr = in_arr;
    arr_abs = abs(in_arr);
    arr_mean = mean(mean(mean(arr_abs)));
    out_arr(arr_abs < (arr_mean))=nan;
end