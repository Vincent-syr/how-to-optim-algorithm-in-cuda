# 1. online_softmax
即softmax的online版本，online版本又进一步细分为，`online_softmax`和`tiling_softmax`. 其主要区别在于每次online计算时是否分块：
- `online_softmax`： 从dim_0~dim_n采用for迭代的形式，对于每个dim，都重新计算`max_val`和`sum_val`
```python
    for i in range(n_dim):
        pre_max = max_val
        max_val = torch.max(max_val, x[i])
        sum_val = sum_val * torch.exp(pre_max - max_val) + torch.exp(x[i] - max_val)
    for i in range(n_dim):
        ans[i] = torch.exp(x[i] - max_val) / sum_val
```

- `tiling_softmax`: 相当于online softmax的并行计算版本。将dim_0~dim_n分为n_split份，每份计算各自的local_max独立的进行softmax的计算；然后再对n_split分，更新的global_max，更新softmax的值; 
```python
    # max(x)
    def m_func(x):
        local_dim = x.shape[-1]
        local_max = torch.tensor(float('-inf'))
        for i in range(local_dim):
            local_max = torch.max(local_max, x[i])
        return local_max

    # 计算softmax的exp(x-max(x)))和sum(exp(x-max(x)))
    def f_func(x, local_max):   
        local_dim = x.shape[-1]
        fx = torch.empty(local_dim)
        local_sum = 0
        for i in range(local_dim):
            fx[i] = torch.exp(x[i] - local_max)
            local_sum += fx[i]
        return fx, local_sum

    # 可并行算
    for i in range(n_split):
        local_x = x[local_dim * i : local_dim * (i+1)]
        local_max[i] = m_func(local_x)  # max_func
        fx[i, :], local_sum[i] = f_func(local_x, local_max[i])
        global_max = torch.max(global_max, local_max[i])

```

# Flash Attention
- Flash Attention
下面是flash attention的pytorch实现版本[flash_attn.py](flash_attn.py).需要根据不同GPU的smem大小，调整分开tiling的size `Br`, `Bc`. 

- Flash Attention V2
[flash_attn2.py](flash_attn2.py). 相比第一版flash attention, 调整了for遍历的顺序，即先遍历`Br`, 在遍历`Bc`


# 参考
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [Flash Attention V1，从硬件到计算逻辑](https://mp.weixin.qq.com/s/gPVwXysJsV2S2FgP3c6Qpw)
- [加速attention计算的工业标准：flash attention 1和2算法的原理及实现](https://blog.csdn.net/bornfree5511/article/details/133657656?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171463777416800178588857%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171463777416800178588857&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-29-133657656-null-null.142^v100^pc_search_result_base4&utm_term=flash%20attention%20pytorch%E5%AE%9E%E7%8E%B0)
