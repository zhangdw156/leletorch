import numpy as np
def linear_regression_gradient_descent(X, y, alpha, iterations):
    m,n=X.shape
    theta=np.zeros((n,1))

    for _ in range(iterations):
        predictions=np.dot(X,theta)
        ers=predictions-y
        grad=(1/m)*X.T.dot(ers)
        theta-=alpha*grad
    return np.round(theta.flatten(),4)

# 主程序
if __name__ == "__main__":
    # 输入矩阵和向量
    matrix_inputx = input()
    array_y = input()
    alpha = input()
    iterations = input()

    # 处理输入
    import ast
    matrix = np.array(ast.literal_eval(matrix_inputx))
    y = np.array(ast.literal_eval(array_y)).reshape(-1,1)
    alpha = float(alpha)
    iterations = int(iterations)

    # 调用函数计算逆矩阵
    output = linear_regression_gradient_descent(matrix,y,alpha,iterations)
    
    # 输出结果
    print(output)

