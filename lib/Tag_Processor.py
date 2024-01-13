import os
import collections
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import scipy as sp

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
    plt.savefig('tag_network.png', format='png', dpi=300, bbox_inches='tight', facecolor=gradio_blue)
    plt.close()
    return 'tag_network.png'

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

def generate_wordcloud(tag_counts):
    wordcloud = WordCloud(width=1600, height=1200, background_color='white')
    wordcloud.generate_from_frequencies(dict(tag_counts))
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('tag_wordcloud.png', format='png')
    plt.close()
    return 'tag_wordcloud.png'

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