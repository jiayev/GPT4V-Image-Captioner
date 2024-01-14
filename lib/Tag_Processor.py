import os
import collections
import random

import matplotlib.pyplot as plt
import networkx as nx

from wordcloud import WordCloud
from itertools import combinations
from lib import Translator


def unique_elements(original, addition):
    original_list = list(map(str.strip, original.split(',')))
    addition_list = list(map(str.strip, addition.split(',')))
    combined_list = []
    seen = set()
    for item in original_list + addition_list:
        if item not in seen and item != '':
            seen.add(item)
            combined_list.append(item)

    return ', '.join(combined_list)

def save_path(folder_path,file_name):
    n_path = os.path.join(folder_path, "Tag_analysis")
    if not os.path.exists(n_path):
        try:
            os.makedirs(n_path)
        except Exception as e:
            print(f"Error : {e}")
    save_path = os.path.join(n_path, file_name)
    return save_path

def modify_tags_in_folder(folder_path, tags_to_remove, tags_to_replace_dict, new_tag, insert_position):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tags = [tag.strip() for tag in content.split(',')]
                # 删除标签
                tags = [tag for tag in tags if tag not in tags_to_remove]
                # 替换标签
                for old_tag, new_tag_replacement in tags_to_replace_dict.items():
                    tags = [new_tag_replacement if tag == old_tag else tag for tag in tags]
                # 添加标签
                if new_tag and new_tag.strip(): 
                    if insert_position == 'Start / 开始':
                        tags.insert(0, new_tag.strip())
                    elif insert_position == 'End / 结束':
                        tags.append(new_tag.strip())
                    elif insert_position == 'Random / 随机':
                        random_index = random.randrange(len(tags)+1)
                        tags.insert(random_index, new_tag.strip())
                
                # 保存修改后的文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    updated_content = ', '.join(tags)
                    f.write(updated_content)
                    
    return "Tags modified successfully."


# 词云
def count_tags_in_folder(folder_path, top_n):
    tags_counter = collections.Counter()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tags = content.split(',')
                    tags = [tag.strip() for tag in tags]
                    tags_counter.update(tags)

    sorted_tags = sorted(tags_counter.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags[:top_n]

def generate_network_graph(folder_path, top_n):
    G = nx.Graph()
    tags_cooccurrence = collections.defaultdict(int)

    # 读取文件并计算标签的共现关系
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tags = list(set(content.split(',')))  # 去重
                    for tag_pair in combinations(tags, 2):
                        if tag_pair[0].strip() and tag_pair[1].strip():  # 确保标签不是空的
                            tags_cooccurrence[tag_pair] += 1

    # 只考虑top n个共现关系
    top_cooccurrences = sorted(tags_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # 添加边到图中
    for (tag1, tag2), weight in top_cooccurrences:
        G.add_edge(tag1.strip(), tag2.strip(), weight=weight)

    # 设置画布大小
    plt.figure(figsize=(24, 12))
    
    # 创建黑色背景
    gradio_blue = '#0B0F19'
    plt.gca().set_facecolor(gradio_blue)

    # 为节点设置大小和颜色
    degrees = dict(G.degree)
    node_size = [v * 100 for v in degrees.values()]
    # 使用更鲜亮的颜色映射
    node_color = [degrees[n] for n in G.nodes]

    # 为边设置宽度
    edge_width = [G[u][v]['weight'] / 100 for u, v in G.edges]  # 除以10是为了使边宽度合适

    # 计算节点的布局
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 绘制节点，使用Plasma配色方案，以适配黑色背景
    nx.draw_networkx_nodes(G, pos, node_size=node_size,
                           node_color=node_color, cmap=plt.cm.plasma, alpha=0.8)

    # 绘制边，使用带有透明度的白色
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.3, edge_color='w')

    # 绘制标签，设置为白色以突出显示
    nx.draw_networkx_labels(G, pos, font_size=12,
                            font_weight='bold', font_color='white',
                            font_family='sans-serif')

    # 移除坐标轴
    plt.axis('off')
    
    # 保存图像
    save_network = save_path(folder_path,'tag_network.png')
    plt.savefig(save_network, format='png', dpi=300, bbox_inches='tight', facecolor=gradio_blue)
    plt.close()
    return save_network

def generate_wordcloud(folder_path, top):
    tag_counts = count_tags_in_folder(folder_path, top)
    wordcloud = WordCloud(width=1600, height=1200, background_color='white')
    wordcloud.generate_from_frequencies(dict(tag_counts))
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    save_wordcloud = save_path(folder_path,'tag_wordcloud.png')
    plt.savefig(save_wordcloud, format='png')
    plt.close()
    return save_wordcloud

# Tag处理
def modify_file_content(file_path, new_content, mode):
    if mode == "skip/跳过" and os.path.exists(file_path):
        print(f"Skip writing, as the file {file_path} already exists.")
        return

    if mode == "overwrite/覆盖" or not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        return

    with open(file_path, 'r+', encoding='utf-8') as file:
        existing_content = file.read()
        file.seek(0)
        if mode == "prepend/前置插入":
            combined_content = unique_elements(new_content, existing_content)
            file.write(combined_content)
            file.truncate()
        elif mode == "append/末尾追加":
            combined_content = unique_elements(existing_content, new_content)
            file.write(combined_content)
            file.truncate()
        else:
            raise ValueError("Invalid mode. Must be 'overwrite/覆盖', 'prepend/前置插入', or 'append/末尾追加'.")

def process_tags(folder_path, top_n, tags_to_remove, tags_to_replace, new_tag, insert_position, translate, api_key,
                 api_url):
    # 解析删除标签
    tags_to_remove_list = tags_to_remove.split(',') if tags_to_remove else []
    tags_to_remove_list = [tag.strip() for tag in tags_to_remove_list]

    # 解析替换标签
    tags_to_replace_dict = {}
    if tags_to_replace:
        try:
            for pair in tags_to_replace.split(','):
                old_tag, new_replacement_tag = pair.split(':')
                tags_to_replace_dict[old_tag.strip()] = new_replacement_tag.strip()
        except ValueError:
            return "Error: Tags to replace must be in 'old_tag:new_tag' format separated by commas", None, None

    # 修改文件夹中的标签
    modify_tags_in_folder(folder_path, tags_to_remove_list, tags_to_replace_dict, new_tag, insert_position)

    # 词云及网格图
    top = int(top_n)
    wordcloud_path = generate_wordcloud(folder_path, top)
    networkgraph_path = generate_network_graph(folder_path, top)

    # 翻译Tag功能
    def truncate_tag(tag, max_length=30): 
        # 截断过长标签
        return (tag[:max_length] + '...') if len(tag) > max_length else tag
    
    tag_counts = count_tags_in_folder(folder_path, top)
    
    if translate.startswith('GPT-3.5 translation / GPT3.5翻译'):
        translator = Translator.GPTTranslator(api_key, api_url)
    elif translate.startswith('Free translation / 免费翻译'):
        translator = Translator.ChineseTranslator()
    else:
        translator = None
    if translator:
        tags_to_translate = [tag for tag, _ in tag_counts]
        translations = Translator.translate_tags(translator, tags_to_translate)
        # 确保 translations 列表长度与 tag_counts 一致
        translations.extend(["" for _ in range(len(tag_counts) - len(translations))])
        tag_counts_with_translation = [(truncate_tag(tag_counts[i][0]), tag_counts[i][1], translations[i]) for i in
                                       range(len(tag_counts))]
    else:
        tag_counts_with_translation = [(truncate_tag(tag), count, "") for tag, count in tag_counts]

    return tag_counts_with_translation, wordcloud_path, networkgraph_path, "Tags processed successfully."