def convert_to_float(input_row):
    output_row = []
    for i in input_row:
        output_row.append(float(i))
    return output_row

def two_d_deep_copy(src_array):
    dest_array = []
    for i in range(0,len(src_array)):
        sub_ra = []
        for j in range(0,len(src_array[i])):
            sub_ra.append(src_array[i][j])
        dest_array.append(sub_ra)
    return dest_array

def total_up_mean_matrix(ra):
    totals_list = []
    for i in range(0,len(ra)):
        total = 0
        for j in range(0,len(ra[i])):
            total += ra[i][j]
        totals_list.append(total)
    return totals_list

def transpose_matrix(matrix):

    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed_matrix = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def matrix_multiply(matrix1, matrix2):
    # Get the dimensions of the matrices
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    # Check if the matrices can be multiplied
    if cols1 != rows2:
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    # Initialize the result matrix with zeros
    result_matrix = [[0] * cols2 for _ in range(rows1)]

    # Perform matrix multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

    return result_matrix
    
def fill_ones(x_ra):
    ret_ra = []
    ones = []
    for i in x_ra:
        ones.append(1)
    ret_ra.append(ones)
    ret_ra.append(x_ra)
    return ret_ra

def build_w(scale1, scale2, length):
    two_d_w_outer = []
    for x in range(0,length):
        row = []
        for y in range(0,length):
            if x == 0 and y == 0:
                row.append(scale1)
            elif x == length-1 and y == length-1:
                row.append(scale1)
            elif x == y:
                row.append(scale2)
            else:
                row.append(0)
        two_d_w_outer.append(row)
    return two_d_w_outer

def find_inverse(matrix):
    try:
        # Check if the matrix is 2x2
        if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
            raise ValueError("Input matrix must be a 2x2 matrix")

        # Get matrix elements
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

        # Calculate the determinant
        determinant = a * d - b * c

        # Check if the matrix is invertible
        if determinant == 0:
            raise ValueError("Matrix is not invertible")

        # Calculate the inverse
        inverse_matrix = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]

        return inverse_matrix

    except Exception as e:
        return "Error: "+str(e)

print("Finding least squares line")
print("(WX)^T*WXß)=(WX)^T*WY")
print("Input all X values of pairs comma separated")
top_row = convert_to_float((input()).split(","))
print("Input all Y values of pairs comma separated")
bottom_row = convert_to_float((input()).split(","))

parent_ra = []
parent_ra.append(top_row)
parent_ra.append(bottom_row)

print("Your X with 1s transposed:")
x_ra = fill_ones(top_row)
x_trans = transpose_matrix(x_ra)
print(x_trans)
print("Your Y transposed:")
y_trans = transpose_matrix([bottom_row])
print(y_trans)
print("Enter scale if one half then 1,2")
scale = convert_to_float((input()).split(","))
print("Finding W")
w_matrix = build_w(scale[0],scale[1],len(y_trans))
print(w_matrix)
print("Any key")
input()
original_ra = two_d_deep_copy(parent_ra)
# find WX
wx_mtrx = matrix_multiply(w_matrix,x_trans)
wy_mtrx = matrix_multiply(w_matrix,y_trans)
print("Finding WX")
print(wx_mtrx)
print("Any key")
input()
print("Finding WY")
print(wy_mtrx)
print("Any key")
input()
wxt_mtrx = transpose_matrix(wx_mtrx)
wxt_wy_mtrx = matrix_multiply(wxt_mtrx,wy_mtrx)
wxt_wx_mtrx = matrix_multiply(wxt_mtrx,wx_mtrx)
print("finding WX^T * WY")
print(wxt_wy_mtrx)
print("finding WX^T * WX")
print(wxt_wx_mtrx)
wxt_wx_invers_mtrx = find_inverse(wxt_wx_mtrx)
print("finding (wx^t*wx) inverse (final result)")
res = matrix_multiply(wxt_wx_invers_mtrx,wxt_wy_mtrx)
print(res)
