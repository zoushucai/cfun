import multiprocessing

from cfun.read import parallel_handle

# 设置 multiprocessing 的启动方式，防止 出现 错误
multiprocessing.set_start_method("spawn", force=True)


def process_item(item, factor, add_value):
    return item * factor + add_value


def test_parallel_handle():
    """
    测试并行处理函数, 由于pytest也是多线程的，所以这里不不能直接使用pytest的多线程测试，需要在开头添加
    multiprocessing.set_start_method("spawn", force=True)

    """
    items = [1, 2, 3, 4]
    factor = 10
    add_value = 5
    args = (factor, add_value)

    # 传递 items, process_item 函数和 args 元组
    results = parallel_handle(items, process_item, args=args)
    print(results)  # 输出 [15, 25, 35, 45] #乱序
