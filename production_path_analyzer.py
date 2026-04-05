import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time


def read_distillation_data(file_path):
    """
    读取石油蒸馏数据
    """
    # 读取Excel文件中的三个工作表
    distillation_df = pd.read_excel(file_path, sheet_name='石油蒸馏数据', header=None)
    cracking_df = pd.read_excel(file_path, sheet_name='裂解数据', header=None)
    post_cracking_df = pd.read_excel(file_path, sheet_name='裂解后蒸馏数据', header=None)

    return distillation_df, cracking_df, post_cracking_df

def parse_distillation_data(distillation_df):
    """
    解析石油蒸馏数据
    返回: {原油类型: {产物类型: 产量}}, product_types, crude_oil_types, {原油类型: 需要时间}
    """
    data = {}
    time_data = {}

    # 提取表头
    product_types = distillation_df.iloc[0, 1:6].tolist()  # B1-F1
    crude_oil_types = distillation_df.iloc[1:5, 0].tolist()  # A2-A5

    # 解析数据
    for i in range(1, 5):  # 行索引1-4对应A2-A5
        crude_oil = distillation_df.iloc[i, 0]
        data[crude_oil] = {}
        time_data[crude_oil] = distillation_df.iloc[i, 6]  # 列6 = 需要时间

        for j in range(1, 6):  # 列索引1-4对应B2-E5
            product = product_types[j - 1]
            quantity = distillation_df.iloc[i, j]
            data[crude_oil][product] = quantity if pd.notna(quantity) else 0

    return data, product_types, crude_oil_types, time_data

def parse_cracking_data(cracking_df):
    """
    解析裂解数据
    返回: {原料: {裂解方式: 产量}}
    """
    data = {}

    # 提取表头
    cracking_methods = cracking_df.iloc[0, 1:8].tolist()  # B1-H1
    feedstocks = cracking_df.iloc[1:7, 0].tolist()  # A2-A7

    # 解析数据
    for i in range(1, 7):  # 行索引1-4对应A2-A7
        feedstock = cracking_df.iloc[i, 0]
        data[feedstock] = {}

        for j in range(1, 8):  # 列索引1-7对应B2-H7
            method = cracking_methods[j - 1]
            quantity = cracking_df.iloc[i, j]
            data[feedstock][method] = quantity / 1000 if pd.notna(quantity) else 0

    return data, cracking_methods, feedstocks

def parse_post_cracking_data(post_cracking_df, cracking_df):
    """
    解析裂解后蒸馏数据
    返回: {(原料, 裂解方式): {产物: 产量}}
    """
    data = {}

    # 提取表头
    product_types = post_cracking_df.iloc[0, 2:22].tolist()  # C1-V1
    feedstocks = cracking_df.iloc[1:5, 0].tolist()  # A2-A37
    cracking_methods = cracking_df.iloc[0, 1:8].tolist()  # B2-B37

    # 解析数据
    for fs in feedstocks:  # 行索引1-36对应A2-A37
        for cm in cracking_methods:
            for i in range(1,37):
                feedstock = post_cracking_df.iloc[i, 0]
                method = post_cracking_df.iloc[i, 1]

                if pd.isna(feedstock) or pd.isna(method):
                    continue

                elif feedstock.__eq__(fs) or method.__eq__(cm):
                    continue

                key = (feedstock, method)
                data[key] = {}

                for k in range(2, 22):  # 列索引2-21对应C2-V37
                    product = product_types[k - 2]
                    quantity = post_cracking_df.iloc[i, k]
                    data[key][product] = quantity / 1000 if pd.notna(quantity) else 0

    return data, product_types, feedstocks, cracking_methods

def get_all_processing_procedures(products, methods, p_m_map):
    data = []

    products.remove("环烷酸")

    products.append("乙烷")
    products.append("丙烷")

    def backtrack(index, current_dict):
        if index == len(products):
            data.append(current_dict.copy())
            return

        if products[index] == "乙烷" or products[index] == "丙烷":
            current_dict[products[index]] = "加氢-重度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-轻度"
            backtrack(index + 1, current_dict)
        elif products[index] == "轻燃油" or products[index] == "石脑油":
            current_dict[products[index]] = "加氢-轻度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "加氢-中度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "加氢-重度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-轻度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-中度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-重度"
            backtrack(index + 1, current_dict)
        else:
            current_dict[products[index]] = "加氢-轻度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "加氢-中度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "加氢-重度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-轻度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-中度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "蒸汽-重度"
            backtrack(index + 1, current_dict)
            current_dict[products[index]] = "直接蒸馏"
            backtrack(index + 1, current_dict)

    backtrack(0, {})
    return data


def calculate_final_products_optimized(distillation_data, cracking_data, post_cracking_data,
                                       product_types, cracking_methods, max_workers=None,
                                       batch_size=4000):
    """
    优化的多线程版本，支持批量处理
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    intermediate_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    processing_procedures = get_all_processing_procedures(product_types, cracking_methods, cracking_data)

    # 如果没有指定最大工作线程数，根据任务数量自动调整
    if max_workers is None:
        max_workers = min(32, (len(processing_procedures) // 10) + 1)
        max_workers = max(4, max_workers)  # 至少4个线程

    print(f"使用 {max_workers} 个工作线程")

    # 对每种原油类型进行计算
    for crude_oil, products in distillation_data.items():
        print(f"\n处理原油类型: {crude_oil}")
        print(f"总裂解组合数: {len(processing_procedures)}")

        # 批量处理以提高效率
        total_batches = (len(processing_procedures) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(processing_procedures))
            batch_procedures = processing_procedures[start_idx:end_idx]

            print(f"处理批次 {batch_idx + 1}/{total_batches} "
                  f"({start_idx}-{end_idx - 1})")

            # 准备批量任务
            tasks = []
            for idx, procedure in enumerate(batch_procedures, start=start_idx):
                task_args = (crude_oil, products, procedure,
                             cracking_data, post_cracking_data, idx)
                tasks.append(task_args)

            # 处理单个批次
            batch_results = process_batch_multi_thread(
                tasks, max_workers, f"批次 {batch_idx + 1}"
            )

            # 合并结果
            for crude, procedure, final_products, intermediate in batch_results:
                for product, qty in final_products.items():
                    results[crude][str(procedure)][product] = qty
                for product, qty in intermediate.items():
                    intermediate_results[crude][str(procedure)][product] = qty

    return results, intermediate_results

def process_batch_multi_thread(tasks, max_workers, batch_name):
    """处理一个批次的任务"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_procedure_with_idx, task_args): task_args
                   for task_args in tasks}

        # 显示进度
        with tqdm(total=len(tasks)+1, desc=batch_name, unit="任务") as pbar:
            pbar.update(1)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task_args = futures[future]
                    crude_oil, products, procedure, cracking_data, post_cracking_data, idx = task_args
                    print(f"\n警告: 任务 {idx} 失败: {e}")
                    # 可以在这里添加重试逻辑或错误记录

                pbar.update(1)

    return results

def process_single_procedure_with_idx(args):
    """
    处理单个裂解组合的函数（用于多线程）
    返回: (原油类型, 裂解方式, 最终产物, 中间产物处理量)
    """
    crude_oil, products, processing_procedure, cracking_data, post_cracking_data, idx = args

    # 初始化最终产物字典
    final_products = defaultdict(float)

    # 记录每种中间产物的总处理量（需要进入裂解的量）
    intermediate_throughput = defaultdict(float)

    # 需要递归处理的中间产物队列
    # 格式: [(产物类型, 初始数量, 裂解方式)]
    queue = []

    # 将蒸馏产物加入队列
    for product, quantity in products.items():
        if quantity > 0:
            if product not in processing_procedure:
                queue.append((product, quantity, ""))
            else:
                queue.append((product, quantity, processing_procedure[product]))

    # 处理队列中的产物，包括递归产生的中间产物
    processed_count = 0
    maximum_acceptable_error_value = 1e-4
    max_iterations = 100000  # 防止无限循环

    while queue and processed_count < max_iterations:
        current_product, current_qty, current_method = queue.pop(0)

        # 检查当前产物是否有裂解数据
        if current_product not in cracking_data:
            # 如果没有裂解数据，直接作为最终产物
            final_products[current_product] += current_qty
            continue

        if current_qty <= maximum_acceptable_error_value:
            continue

        # 记录中间产物处理量
        intermediate_throughput[current_product] += current_qty

        # 获取当前产物在选定裂解方式下的裂解产量
        cracking_yield = cracking_data[current_product].get(current_method, 0)

        # 计算裂解后得到的中间产物量
        intermediate_qty = current_qty * cracking_yield

        # 获取裂解后蒸馏数据
        key = (current_product, current_method)
        if key in post_cracking_data:
            # 根据裂解后蒸馏数据分配产物
            for output_product, ratio in post_cracking_data[key].items():
                output_qty = intermediate_qty * ratio

                if output_qty > 0:
                    # 检查产物是否需要再次裂解
                    if output_product in cracking_data:
                        # 中间产物，需要再次裂解
                        queue.append((output_product, output_qty, processing_procedure[output_product]))
                    else:
                        # 最终产物
                        final_products[output_product] += output_qty
        processed_count += 1

    return crude_oil, processing_procedure, final_products, intermediate_throughput

# def calculate_final_products(distillation_data, cracking_data, post_cracking_data,
#                              product_types, cracking_methods):
#     """
#     计算最终产物
#     处理需要递归裂解的情况
#     """
#     results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#
#     processing_procedures = get_all_processing_procedures(product_types, cracking_methods, cracking_data)
#
#     # 对每种原油类型进行计算
#     for crude_oil, products in distillation_data.items():
#         print(f"\n处理原油类型: {crude_oil}")
#
#         # 对每种裂解组合方式进行计算
#         for processing_procedure in tqdm(processing_procedures):
#             # 初始化最终产物字典
#             final_products = defaultdict(float)
#
#             # 需要递归处理的中间产物队列
#             # 格式: [(产物类型, 初始数量, 裂解方式)]
#             queue = []
#
#             # 将蒸馏产物加入队列
#             for product, quantity in products.items():
#                 if quantity > 0:
#                     if product not in processing_procedure:
#                         queue.append((product, quantity, ""))
#                     else:
#                         queue.append((product, quantity, processing_procedure[product]))
#
#             # 处理队列中的产物，包括递归产生的中间产物
#             processed_count = 0
#             maximum_acceptable_error_value = 1e-4
#             max_iterations = 10000  # 防止无限循环
#             while queue and processed_count < max_iterations:
#                 current_product, current_qty, current_method = queue.pop(0)
#                 # 检查当前产物是否有裂解数据
#                 if current_product not in cracking_data:
#                     # 如果没有裂解数据，直接作为最终产物
#                     final_products[current_product] += current_qty
#                     continue
#
#                 if current_qty <= maximum_acceptable_error_value:
#                     continue
#
#                 # 获取当前产物在选定裂解方式下的裂解产量
#                 cracking_yield = cracking_data[current_product].get(current_method, 0)
#
#                 # 计算裂解后得到的中间产物量
#                 intermediate_qty = current_qty * cracking_yield
#
#                 # 获取裂解后蒸馏数据
#                 key = (current_product, current_method)
#                 if key in post_cracking_data:
#                     # 根据裂解后蒸馏数据分配产物
#                     for output_product, ratio in post_cracking_data[key].items():
#                         output_qty = intermediate_qty * ratio
#
#                         if output_qty > 0:
#                             # 检查产物是否需要再次裂解
#                             if output_product in cracking_data:
#                                 # 中间产物，需要再次裂解
#                                 queue.append((output_product, output_qty, processing_procedure[output_product]))
#                             else:
#                                 # 最终产物
#                                 final_products[output_product] += output_qty
#                 else:
#                     print(f"警告: 未找到裂解后蒸馏数据: {key}")
#                 processed_count += 1
#
#                 if processed_count >= max_iterations:
#                     print(f"警告: 达到最大迭代次数 {max_iterations}")
#
#             # 保存结果
#             for product, qty in final_products.items():
#                 results[crude_oil][str(processing_procedure)][product] = qty
#
#     return results


def _add_group_header(ws, group_ranges):
    """
    在第1行添加合并单元格的分组表头
    group_ranges: [(分组名, 起始列号, 结束列号), ...]  列号从1开始
    """
    from openpyxl.styles import Alignment
    for name, start_col, end_col in group_ranges:
        ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)
        cell = ws.cell(row=1, column=start_col, value=name)
        cell.alignment = Alignment(horizontal='center')


def save_results_to_excel(results, intermediate_results, time_data, output_file):
    """
    将结果保存到Excel文件
    - 裂解方式拆分为多列（重燃油、轻燃油、石脑油、炼油气、乙烷、丙烷）
    - 每秒产出表中合并中间产物处理量
    - 上方增加合并单元格的分组表头行
    """
    import ast

    procedure_col_names = ['重燃油', '轻燃油', '石脑油', '炼油气', '乙烷', '丙烷']

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # ===== Sheet 1: 每1000L产出 =====
        rows = []
        for crude_oil, methods in results.items():
            for method_str, products in methods.items():
                procedure = ast.literal_eval(method_str)
                row = {'原油类型': crude_oil}
                for col in procedure_col_names:
                    row[col] = procedure.get(col, '')
                for product, qty in products.items():
                    if qty > 0:
                        row[product] = qty
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            meta_cols = ['原油类型']
            product_cols = sorted([c for c in df.columns
                                   if c not in meta_cols + procedure_col_names])
            df = df[meta_cols + procedure_col_names + product_cols]

            # startrow=1 留出第1行给分组表头
            df.to_excel(writer, sheet_name='每1000L产出', index=False, startrow=1)
            ws = writer.sheets['每1000L产出']

            # 裂解方式分组表头：列从 meta_cols 之后开始
            proc_start = len(meta_cols) + 1
            proc_end = proc_start + len(procedure_col_names) - 1
            _add_group_header(ws, [('裂解方式', proc_start, proc_end)])

            print(f"每1000L产出: {len(df)} 条记录")

        # ===== Sheet 2: 每秒产出 + 中间产物处理量 =====
        rows = []
        for crude_oil, methods in results.items():
            t = time_data[crude_oil]
            for method_str, products in methods.items():
                procedure = ast.literal_eval(method_str)
                row = {'原油类型': crude_oil, '蒸馏时间(s)': t}
                for col in procedure_col_names:
                    row[col] = procedure.get(col, '')
                for product, qty in products.items():
                    if qty > 0:
                        row[product] = qty / t
                # 中间产物处理量，加前缀避免与裂解方式列重名
                for product, qty in intermediate_results[crude_oil][method_str].items():
                    if qty > 0:
                        row['中间_' + product] = qty / t
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            meta_cols = ['原油类型', '蒸馏时间(s)']
            product_cols = sorted([c for c in df.columns
                                   if c not in meta_cols + procedure_col_names
                                   and not c.startswith('中间_')])
            inter_cols = sorted([c for c in df.columns if c.startswith('中间_')])

            df = df[meta_cols + procedure_col_names + product_cols + inter_cols]

            # 写入数据前，将中间产物列名的前缀去掉（只在表头显示原名）
            rename_map = {c: c.replace('中间_', '') for c in inter_cols}
            df = df.rename(columns=rename_map)

            df.to_excel(writer, sheet_name='每秒产出', index=False, startrow=1)
            ws = writer.sheets['每秒产出']

            # 裂解方式分组表头
            proc_start = len(meta_cols) + 1
            proc_end = proc_start + len(procedure_col_names) - 1
            # 中间产物处理量分组表头
            inter_start = len(meta_cols) + len(procedure_col_names) + len(product_cols) + 1
            inter_end = inter_start + len(inter_cols) - 1

            groups = [('裂解方式', proc_start, proc_end)]
            if inter_cols:
                groups.append(('每秒中间产物处理量', inter_start, inter_end))
            _add_group_header(ws, groups)

            print(f"每秒产出: {len(df)} 条记录")

        print(f"\n结果已保存到: {output_file}")


def main():
    # 输入文件路径
    input_file = "蒸馏数据.xlsx"
    output_file = "蒸馏计算结果.xlsx"

    try:
        # 1. 读取数据
        print("正在读取数据...")
        distillation_df, cracking_df, post_cracking_df = read_distillation_data(input_file)

        # 2. 解析数据
        print("正在解析石油蒸馏数据...")
        distillation_data, product_types, crude_oil_types, time_data = parse_distillation_data(distillation_df)

        print("正在解析裂解数据...")
        cracking_data, cracking_methods, feedstocks = parse_cracking_data(cracking_df)

        print("正在解析裂解后蒸馏数据...")
        post_cracking_data, post_cracking_products, post_feedstocks, post_methods = parse_post_cracking_data(
            post_cracking_df, cracking_df)

        # 打印基本信息
        print(f"\n数据解析完成:")
        print(f"- 原油类型: {crude_oil_types}")
        print(f"- 初始产物类型: {product_types}")
        print(f"- 裂解方式: {cracking_methods}")
        print(f"- 裂解原料: {feedstocks}")
        print(f"- 裂解后产物类型: {len(post_cracking_products)} 种")

        # 3. 计算最终产物
        print("\n正在计算最终产物...")
        results, intermediate_results = calculate_final_products_optimized(
            distillation_data,
            cracking_data,
            post_cracking_data,
            product_types,
            cracking_methods
        )

        # 4. 保存结果
        print("\n正在保存结果...")
        save_results_to_excel(results, intermediate_results, time_data, output_file)

        # 5. 显示部分结果
        print("\n部分计算结果示例:")
        for i, (crude_oil, methods) in enumerate(list(results.items())[:2]):  # 只显示前2种原油
            print(f"\n{crude_oil}:")
            for j, (method, products) in enumerate(list(methods.items())[:2]):  # 只显示前2种裂解方式
                print(f"  {method}:")
                for product, qty in list(products.items())[:3]:  # 只显示前3种产物
                    if qty > 0:
                        print(f"    {product}: {qty:.2f}")

        print(f"\n处理完成! 请查看输出文件: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()