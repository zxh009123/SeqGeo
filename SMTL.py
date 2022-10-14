# import tensorflow as tf
# sess = tf.compat.v1.Session()
# with sess.as_default():
#     # tensor = tf.random.uniform([3,4])
#     # tensor = tf.constant([[1,2,3,4],[1,2,3,4],[1,2,3,4]], dtype=float)
#     tensor = tf.random.uniform([3,2])
#     b = tf.random.uniform([3,2])
#     dist_array = 2 - 2 * tf.matmul(tensor, b, transpose_b=True)
#     print_op1 = tf.print("dist_array:", dist_array)
#     pos_dist = tf.diag_part(dist_array)
#     triplet_dist_g2s = pos_dist - dist_array
#     print_op2 = tf.print("triplet_dist_g2s:",triplet_dist_g2s)
#     triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
#     print_op3 = tf.print("triplet_dist_s2g:",triplet_dist_s2g)
#     with tf.control_dependencies([print_op1, print_op2, print_op3]):
#         out = tf.add(triplet_dist_s2g, triplet_dist_g2s)
# sess.run(out)

# import tensorflow as tf

# v1 = tf.constant([1,2,8], dtype=float)
# # v1 = tf.expand_dims(v1, 1)
# v2 = tf.constant([[1,2,3],[1,2,3],[1,2,3]], dtype=float)
# diff = v1 - v2 
# loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(diff * 10.0)))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(diff.eval()) # <-- `diff` contains only 'ones' because of broadcasting
#     print(loss_s2g.eval())

# # [[[[1. 1. 1.]
# #    [1. 1. 1.]]
# # 
# #   [[1. 1. 1.]
# #    [1. 1. 1.]]]]

# print(diff.get_shape().as_list()) # [1, 2, 2, 3] <-- same shape as `v1`

import torch 
import numpy as np

def softMarginTripletLoss(sat_feature, grd_feature, gamma):
    batch_size = sat_feature.shape[0]
    dist_array = 2 - 2 * torch.matmul(sat_feature, torch.transpose(grd_feature, 0, 1))
    pos_dist = torch.diagonal(dist_array)
    
    pair_n = batch_size * (batch_size - 1.0)
    
    #ground to satellite
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * gamma))) / pair_n

    # satellite to ground
    pos_dist_s2g = pos_dist.unsqueeze(0).permute(1,0)
    triplet_dist_s2g  = pos_dist_s2g - dist_array
    loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * gamma))) / pair_n

    return (loss_s2g + loss_g2s) / 2.0

def soft_margin_triplet_loss(sate_vecs, pano_vecs, loss_weight=10, hard_topk_ratio=1.0):
    dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    # Match from satellite to street pano
    triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
    loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
    loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_s2p = loss_s2p.view(-1)
        loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
    loss_s2p = loss_s2p.sum() / num_hard_triplets

    # Match from street pano to satellite
    triplet_dist_p2s = pos_dists - dists
    loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
    loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_p2s = loss_p2s.view(-1)
        loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / num_hard_triplets
    # Total loss
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss

if __name__ == "__main__":
    v1 = torch.rand(4,512)
    t = torch.linalg.norm(v1,dim=1,keepdim=True)
    v1 = v1 / torch.linalg.norm(v1,dim=1,keepdim=True)
    v2 = torch.rand(4,512)
    v2 = v2 / torch.linalg.norm(v2,dim=1,keepdim=True)
    print(soft_margin_triplet_loss(v1, v2, 10.0))
    print(softMarginTripletLoss(v1, v2, 10.0))




# import torch
# v1 = torch.tensor([1,2,8])
# v1 = v1.unsqueeze(0).permute(1,0)
# print(v1.shape)
# v2 = torch.tensor([[1,2,3],[1,2,3],[1,2,3]])
# diff = v1 - v2 
# print(diff)
# loss_g2s = torch.sum(torch.log(1 + torch.exp(diff * 10.0)))
# print(loss_g2s)
