import sys

image=eval(sys.stdin.readline().strip())
kernel=eval(sys.stdin.readline().strip())
params=eval(sys.stdin.readline().strip())


i,j,filter_width,filter_height,padding_size,stride=params

image_2d=image[0]
kernel_2d=kernel[0]

conv_result=0
for row in range(filter_height):
    for col in range(filter_width):
        img_val=image_2d[i+row][j+col]
        kernel_val=kernel_2d[row][col]
        conv_result+=img_val*kernel_val
input_size=len(image_2d)
kernel_size=filter_width
output_size=(input_size+2*padding_size-kernel_size)//stride+1
print(conv_result,output_size)