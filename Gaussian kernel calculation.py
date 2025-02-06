import numpy as np
import pandas as pd


def GIP_kernel(Asso_RNA_Dis):
    def getGosiR(Asso_RNA_Dis):
        # calculate the r in GOsi Kerel
        nc = Asso_RNA_Dis.shape[0]
        summ = 0
        for i in range(nc):
            x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
            x_norm = np.square(x_norm)
            summ = summ + x_norm
        r = summ / nc
        return r
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix



if __name__ == '__main__':

    A = np.array(pd.read_csv('./data/circ-dis.csv', header=None,index_col=None))
    GIP_mr_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    pd.DataFrame(GIP_mr_sim).to_csv('./data/circ4.csv',header=None,index=None)
    pd.DataFrame(GIP_d_sim).to_csv('./data/dis.csv',header=None,index=None)



