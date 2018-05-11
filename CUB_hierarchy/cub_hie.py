import numpy as np
import scipy.io as scio
import pickle


class node:
    def __init__(self, name, attribute, seen=True):
        self.name = name
        self.attribute = attribute
        self.father = None
        self.children = []
        self.attention_vector = np.zeros(312)
        self.knowledge = np.zeros((72, 312))
        self.mask = 0
        self.seen = seen
        self.seen_mask = np.zeros(72)
        self.unseen_mask = np.zeros(72)

    def set_father(self, father):
        self.father = father

    def add_children(self, kids):
        self.children.append(kids)
        if kids.seen:
            self.seen_mask[self.mask] = 1.0
        else:
            self.unseen_mask[self.mask] = 1.0
        self.knowledge[self.mask, :] = kids.attribute
        self.mask += 1


att = scio.loadmat(open('att_splits.mat'))
train_class_list = []
with open('trainvalclasses.txt') as train_class:
    for i in train_class.readlines():
        train_class_list.append(i[:-1])

test_class_list = []
with open('testclasses.txt') as test_classes:
    for i in test_classes.readlines():
        test_class_list.append(i[:-1])


dic = {}
for i in xrange(200):
    dic[int(att['allclasses_names'][i][0][0][0:3])] = att['att'][:, i]

all_class = {}
for i in xrange(200):
    if att['allclasses_names'][i][0][0] in train_class_list:
        seen = True
    else:
        seen = False
    temp = node(att['allclasses_names'][i][0][0], dic[int(att['allclasses_names'][i][0][0][0:3])], seen=seen)
    all_class[int(att['allclasses_names'][i][0][0][0:3])] = temp

#for i in xrange(1,201):
#    print(all_class[i].seen)

high_layer = []
#Albatross
attribute = np.mean([dic[i] for i in range(1, 4)], axis=0)
temp = node('Albatross', attribute)
high_layer.append(temp)
for i in range(1, 4):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

#Auklet
attribute = np.mean([dic[i] for i in range(5, 9)], axis=0)
temp = node('Auklet', attribute)
high_layer.append(temp)
for i in range(5, 9):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(9, 13)], axis=0)
temp = node('Blackbird', attribute)
high_layer.append(temp)
for i in range(9, 13):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(14, 17)], axis=0)
temp = node('Bunting', attribute)
high_layer.append(temp)
for i in range(14, 17):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(18, 20)], axis=0)
temp = node('Catbird', attribute)
high_layer.append(temp)
for i in range(18, 20):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(23, 26)], axis=0)
temp = node('Cormorant', attribute)
high_layer.append(temp)
for i in range(23, 26):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(26, 28)], axis=0)
temp = node('Cowbird', attribute)
high_layer.append(temp)
for i in range(26, 28):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(29, 31)], axis=0)
temp = node('Crow', attribute)
high_layer.append(temp)
for i in range(29, 31):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(31, 34)], axis=0)
temp = node('Cuckoo', attribute)
high_layer.append(temp)
for i in range(31, 34):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(34, 36)], axis=0)
temp = node('Finch', attribute)
high_layer.append(temp)
for i in range(34, 36):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(37, 44)], axis=0)
temp = node('Flycatcher', attribute)
high_layer.append(temp)
for i in range(37, 44):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(47, 49)], axis=0)
temp = node('Goldfinch', attribute)
high_layer.append(temp)
for i in range(47, 49):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(50, 54)], axis=0)
temp = node('Grebe', attribute)
high_layer.append(temp)
for i in range(50, 54):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(54, 58)], axis=0)
temp = node('Grosbeak', attribute)
high_layer.append(temp)
for i in range(54, 58):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(59, 67)], axis=0)
temp = node('Gull', attribute)
high_layer.append(temp)
for i in range(59, 67):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(67, 70)], axis=0)
temp = node('Hummingbird', attribute)
high_layer.append(temp)
for i in range(67, 70):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(71, 73)], axis=0)
temp = node('Jaeger', attribute)
high_layer.append(temp)
for i in range(71, 73):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(73, 76)], axis=0)
temp = node('Jay', attribute)
high_layer.append(temp)
for i in range(73, 76):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(77, 79)], axis=0)
temp = node('Kingbird', attribute)
high_layer.append(temp)
for i in range(77, 79):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(79, 84)], axis=0)
temp = node('Kingfisher', attribute)
high_layer.append(temp)
for i in range(79, 84):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(89, 91)], axis=0)
temp = node('Merganser', attribute)
high_layer.append(temp)
for i in range(89, 91):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(95, 99)], axis=0)
temp = node('Oriole', attribute)
high_layer.append(temp)
for i in range(95, 99):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(100, 102)], axis=0)
temp = node('Pelican', attribute)
high_layer.append(temp)
for i in range(100, 102):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(111, 113)], axis=0)
temp = node('Shrike', attribute)
high_layer.append(temp)
for i in range(111, 113):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(113, 134)], axis=0)
temp = node('Sparrow', attribute)
high_layer.append(temp)
for i in range(113, 134):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(135, 139)], axis=0)
temp = node('Swallow', attribute)
high_layer.append(temp)
for i in range(135, 139):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(139, 141)], axis=0)
temp = node('Tanager', attribute)
high_layer.append(temp)
for i in range(139, 141):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(141, 148)], axis=0)
temp = node('Tern', attribute)
high_layer.append(temp)
for i in range(141, 148):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(149, 151)], axis=0)
temp = node('Thrasher', attribute)
high_layer.append(temp)
for i in range(149, 151):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(151, 158)], axis=0)
temp = node('Vireo', attribute)
high_layer.append(temp)
for i in range(151, 158):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(158, 183)], axis=0)
temp = node('Warbler', attribute)
high_layer.append(temp)
for i in range(158, 183):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(183, 185)], axis=0)
temp = node('Waterthrush', attribute)
high_layer.append(temp)
for i in range(183, 185):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(185, 187)], axis=0)
temp = node('Waxwing', attribute)
high_layer.append(temp)
for i in range(185, 187):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(187, 193)], axis=0)
temp = node('Woodpecker', attribute)
high_layer.append(temp)
for i in range(187, 193):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(193, 200)], axis=0)
temp = node('Wren', attribute)
high_layer.append(temp)
for i in range(193, 200):
    all_class[i].set_father(temp)
    temp.add_children(all_class[i])

attribute = np.mean([dic[i] for i in range(1, 201)], axis=0)
temp = node('bird', attribute)
print(len(temp.children))
count = 0
for i in range(1, 201):
    if all_class[i].father is None:
        all_class[i].set_father(temp)
        temp.add_children(all_class[i])
        count += 1

for i in high_layer:
    i.set_father(temp)
    temp.add_children(i)

root = temp
knowledge_dict = {}
attention_dict = {}


def depth_search(nnode):
    if len(nnode.children) == 0:
        return
    knowledge_dict[nnode.name] = nnode.knowledge
    atten = np.zeros((len(nnode.children), 312))
    for (i, ch) in enumerate(nnode.children, 0):
        atten[i, :] = ch.attribute
    attention_dict[nnode.name] = np.max(atten, axis=0) - np.min(atten, axis=0)
    for i in nnode.children:
        depth_search(i)


depth_search(root)
all_data = []
with open('images.txt', 'r') as image_file:
    for line in image_file.readlines():
        data = {}
        i_id, i_name = line.split(' ')
        cls_id = int(i_name.split('.')[0])
        data['name'] = i_name[:-1]
        nnode = all_class[cls_id]
        if nnode.father.name == 'bird':
            data['layer_coarse'] = {}
            data['layer_coarse']['father'] = nnode.father.name
            data['layer_coarse']['mask'] = nnode.father.mask
            data['layer_coarse']['seen_mask'] = nnode.father.seen_mask
            data['layer_coarse']['unseen_mask'] = nnode.father.unseen_mask
            data['layer_coarse']['label'] = nnode.father.children.index(nnode)
        else:
            data['layer_fine'] = {}
            data['layer_fine']['father'] = nnode.father.name
            data['layer_fine']['mask'] = nnode.father.mask
            data['layer_fine']['seen_mask'] = nnode.father.seen_mask
            data['layer_fine']['unseen_mask'] = nnode.father.unseen_mask
            data['layer_fine']['label'] = nnode.father.children.index(nnode)

            nnode = nnode.father
            data['layer_coarse'] = {}
            data['layer_coarse']['father'] = nnode.father.name
            data['layer_coarse']['mask'] = nnode.father.mask
            data['layer_coarse']['seen_mask'] = nnode.father.seen_mask
            data['layer_coarse']['unseen_mask'] = nnode.father.unseen_mask
            data['layer_coarse']['label'] = nnode.father.children.index(nnode)
        all_data.append(data)


train_image_id = open('trainimages.txt', 'wb')
test_image_id = open('testimages.txt', 'wb')
for i in xrange(len(all_data)):
    data = all_data[i]
    i_name = data['name']
    cls_name = i_name.split('/')[0]
    if cls_name in train_class_list:
        train_image_id.write(str(i) + ' ' + i_name + '\n')
    else:
        test_image_id.write(str(i) + ' ' + i_name + '\n')
train_image_id.close()
test_image_id.close()

pickle.dump(all_data, open('cub_image_dict.pkl', 'wb'))
pickle.dump(knowledge_dict, open('cub_knowledge_dict.pkl', 'wb'))
pickle.dump(attention_dict, open('cub_attention_dict.pkl', 'wb'))

