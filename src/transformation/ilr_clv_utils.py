import numpy as np
import tensorflow as tf

EPS = 1e-6

class Tree(object):
    def __init__(self, parent=None, left=None, right=None,
                 node_idx=None, inode_idx=None, taxon_idx=None,
                 b=0.0, depth=0):
        self.left = left
        self.right = right
        self.parent = parent
        self.inode_idx = inode_idx
        self.taxon_idx = taxon_idx
        self.node_idx = node_idx
        if inode_idx is not None:
            self.b = tf.sigmoid(tf.Variable(b, dtype=tf.float32))
        elif taxon_idx is not None:
            self.b = b
        else:
            raise ValueError("the node must either be an inode or a leaf (taxon)")
        self.depth = depth

    def is_taxon(self):
        return self.left is None and self.right is None

    def print_tree(self, level=0):
        name = self.name()
        print("   " * level + name)
        if self.left is not None:
            self.left.print_tree(level + 1)
        if self.right is not None:
            self.right.print_tree(level + 1)

    def name(self):
        if self.is_taxon():
            name = "node {} taxon {}".format(self.node_idx, self.taxon_idx)
        else:
            name = "node {} inode {}".format(self.node_idx, self.inode_idx)
        return name


def convert_theta_to_tree(theta):
    n_node = theta.shape[0] + theta.shape[1]
    node_reference = [0 for _ in range(n_node)]  # placeholder
    root = convert_theta_to_tree_helper(theta, node_reference, None)
    return root, node_reference


def convert_theta_to_tree_helper(theta, node_reference, parent, is_left_child=True):
    # find and return parent's left/right child node
    if parent is None:
        n_taxa = theta.shape[1]
        node_taxa = np.arange(n_taxa)
    else:
        parent_taxon_idx = parent.inode_idx
        assert parent_taxon_idx is not None
        theta_node = theta[parent_taxon_idx]
        if is_left_child:
            node_taxa = np.where(theta_node == +1)[0]
        else:
            node_taxa = np.where(theta_node == -1)[0]

    depth = 0 if parent is None else parent.depth + 1
    if len(node_taxa) == 1:
        taxon_idx = node_taxa[0]
        n_inode = theta.shape[0]
        node_idx = n_inode + taxon_idx
        child = Tree(parent=parent, taxon_idx=taxon_idx, node_idx=node_idx,
                     b=0.0, depth=depth)  # taxon nodes cannot break
        node_reference[node_idx] = child
    else:
        inode_idx = -1
        for i, theta_i in enumerate(theta):
            leaves = np.where(theta_i != 0)[0]
            if len(leaves) == len(node_taxa) and (leaves == node_taxa).all():
                inode_idx = i
                break
        assert inode_idx != -1, "cannot find the child whose leaves should be {}".format(node_taxa)

        child = Tree(parent=parent, inode_idx=inode_idx, node_idx=inode_idx, b=0.0, depth=depth)
        child.left = convert_theta_to_tree_helper(theta, node_reference, child, is_left_child=True)
        child.right = convert_theta_to_tree_helper(theta, node_reference, child, is_left_child=False)
        node_reference[inode_idx] = child

    return child

# ------------------------------------------ between-group interaction ------------------------------------------ #


def get_p_i(n_inode, root):
    p_i = [0 for _ in range(n_inode)]
    get_p_i_helper(p_i, root, 1)
    p_i = tf.stack(p_i)
    return p_i


def get_p_i_helper(p_i, node, p_ancestors_break):
    if node.is_taxon():
        return
    node_idx = node.node_idx
    p_ancestors_break *= node.b
    p_i[node_idx] = p_ancestors_break
    get_p_i_helper(p_i, node.left, p_ancestors_break)
    get_p_i_helper(p_i, node.right, p_ancestors_break)


def get_between_group_p_j_given_i(n_node, n_inode, root):
    p_j_given_i = [0 for _ in range(n_inode)]
    get_between_group_p_j_given_i_helper(p_j_given_i, n_node, root, root)
    p_j_given_i = tf.stack(p_j_given_i, axis=0)
    return p_j_given_i


def get_between_group_p_j_given_i_helper(p_j_given_i, n_node, root, node):
    # node: inode i
    if node.is_taxon():
        return
    b_copy = node.b
    node.b = 1.0

    p_j_given_inode = [0 for _ in range(n_node)]
    get_between_group_p_j_given_inode_helper(p_j_given_inode, root, 1)
    get_between_group_p_j_given_i_helper(p_j_given_i, n_node, root, node.left)
    get_between_group_p_j_given_i_helper(p_j_given_i, n_node, root, node.right)
    p_j_given_inode = tf.stack(p_j_given_inode, axis=0)

    inode_idx = node.inode_idx
    p_j_given_i[inode_idx] = p_j_given_inode
    node.b = b_copy


def get_between_group_p_j_given_inode_helper(p_j_given_inode, node, p_ancestors_break):
    if node is None:
        return
    node_idx = node.node_idx
    p_j_given_inode[node_idx] = p_ancestors_break * (1.0 - node.b)
    get_between_group_p_j_given_inode_helper(p_j_given_inode, node.left, p_ancestors_break * node.b)
    get_between_group_p_j_given_inode_helper(p_j_given_inode, node.right, p_ancestors_break * node.b)


# --------------------------------------------- in-group interaction -------------------------------------------- #


def get_in_group_p_j_given_i(n_node, n_inode, root):
    p_j_given_i = [0 for _ in range(n_inode)]
    get_in_group_p_j_given_i_initialization(root)
    get_in_group_p_j_given_i_helper(p_j_given_i, n_node, root, root)
    get_in_group_p_j_given_i_clean(root)
    p_j_given_i = tf.stack(p_j_given_i, axis=0)
    return p_j_given_i


def get_in_group_p_j_given_i_initialization(root):
    def helper(node):
        node.is_ancestor_of_i = False
        if not node.is_taxon():
            helper(node.left)
            helper(node.right)
    helper(root)


def get_in_group_p_j_given_i_clean(root):
    def helper(node):
        del node.is_ancestor_of_i
        if not node.is_taxon():
            helper(node.left)
            helper(node.right)
    helper(root)


def get_in_group_p_j_given_i_helper(p_j_given_i, n_node, root, node):
    # node: inode i
    if node.is_taxon():
        return

    node.is_ancestor_of_i = True  # tmp value
    p_j_given_inode = [0 for _ in range(n_node)]
    get_in_group_p_j_given_inode_helper(p_j_given_inode, root, 1)
    p_j_given_inode = tf.stack(p_j_given_inode, axis=0)
    inode_idx = node.inode_idx
    p_j_given_i[inode_idx] = p_j_given_inode

    get_in_group_p_j_given_i_helper(p_j_given_i, n_node, root, node.left)
    get_in_group_p_j_given_i_helper(p_j_given_i, n_node, root, node.right)

    node.is_ancestor_of_i = False


def get_in_group_p_j_given_inode_helper(p_j_given_inode, node, p_common_ancestors_break):
    node_idx = node.node_idx
    if node.is_taxon():
        p_j_given_inode[node_idx] = 1.0 - p_common_ancestors_break
    else:
        p_j_given_inode[node_idx] = 0.0
        if node.is_ancestor_of_i:
            p_common_ancestors_break *= node.b
        get_in_group_p_j_given_inode_helper(p_j_given_inode, node.left, p_common_ancestors_break)
        get_in_group_p_j_given_inode_helper(p_j_given_inode, node.right, p_common_ancestors_break)


# --------------------------------------------------- dynamics -------------------------------------------------- #


def get_inode_relative_abundance(root, x_t, n_inode):
    r_t_inode = [0 for _ in range(n_inode)]
    get_inode_relative_abundance_helper(r_t_inode, root, x_t)
    r_t_inode = tf.stack(r_t_inode, axis=-1)
    return r_t_inode


def get_inode_relative_abundance_helper(r_t_inode, node, x_t):
    if node.is_taxon():
        return x_t[..., node.taxon_idx]
    inode_idx = node.inode_idx
    left_r_t = get_inode_relative_abundance_helper(r_t_inode, node.left, x_t)
    right_r_t = get_inode_relative_abundance_helper(r_t_inode, node.right, x_t)
    inode_r_t = left_r_t + right_r_t
    r_t_inode[inode_idx] = inode_r_t
    return inode_r_t


# -------------------------------------------- ilr and inverse ilr ---------------------------------------------- #


def get_n_plus_and_n_minus(theta):
    Dm1, D = theta.shape
    w = np.ones(D)
    n_plus = np.empty(Dm1, dtype=np.float)
    n_minus = np.empty(Dm1, dtype=np.float)
    for i in range(Dm1):
        n_plus[i] = np.sum(w[theta[i] == 1])
        n_minus[i] = np.sum(w[theta[i] == -1])
    return n_plus, n_minus


def get_psi(theta):
    m, n = theta.shape
    n_plus, n_minus = get_n_plus_and_n_minus(theta)
    psi = np.zeros_like(theta, dtype=np.float)
    for i in range(m):
        for j in range(n):
            if theta[i, j] == 1:
                psi[i, j] = 1 / n_plus[i] * np.sqrt(n_plus[i] * n_minus[i] / (n_plus[i] + n_minus[i]))
            elif theta[i, j] == -1:
                psi[i, j] = -1 / n_minus[i] * np.sqrt(n_plus[i] * n_minus[i] / (n_plus[i] + n_minus[i]))
    return psi


def inverse_ilr_transform(ystar, psi):
    """
    ystar: (D-1,), psi: (D-1, D), w: (D, )
    return: x: (D,)
    """

    # (1, D-1) * (D-1, D) -> (1, D)
    exp_ystar_psi = tf.exp(tf.reduce_sum(ystar[..., None] * psi, axis=-2))

    y = exp_ystar_psi / tf.reduce_sum(exp_ystar_psi, axis=-1, keepdims=True)
    w = tf.ones_like(y)
    x = y * w / tf.reduce_sum(y * w, axis=-1, keepdims=True)
    return x


def log_geometric_mean(y, w):
    return tf.reduce_sum(w * tf.log(y + EPS), axis=-1) / tf.reduce_sum(w, axis=-1)


def ilr_transform(x, theta):
    """
    x: (bsz, D, ), w: (D, ), theta: (D-1, D), n_plus: (D-1, ), n_minuse: (D-1, )
    return ystar: (D-1, )
    """
    D = theta.shape[1]
    n_plus, n_minus = get_n_plus_and_n_minus(theta)
    w = tf.ones_like(x)
    y = x / w

    # compute log geometric mean ratio
    log_gm_ratio = [0 for _ in range(D - 1)]
    for i in range(D-1):
        plus_idx = [idx for idx in range(D) if theta[i][idx] == 1]
        y_plus = tf.stack([y[..., idx] for idx in plus_idx], axis=-1)
        w_plus = tf.stack([w[..., idx] for idx in plus_idx], axis=-1)
        loggp_yi_plus = log_geometric_mean(y_plus, w_plus)

        minus_idx = [idx for idx in range(D) if theta[i][idx] == -1]
        y_minus = tf.stack([y[..., idx] for idx in minus_idx], axis=-1)
        w_minus = tf.stack([w[..., idx] for idx in minus_idx], axis=-1)
        loggp_yi_minus = log_geometric_mean(y_minus, w_minus)

        log_gm_ratio[i] = loggp_yi_plus - loggp_yi_minus
    log_gm_ratio = tf.stack(log_gm_ratio, axis=-1)

    normalizing_constant = np.sqrt(n_plus * n_minus / (n_plus + n_minus))  # (D-1, )

    ystar = normalizing_constant * log_gm_ratio
    return ystar
