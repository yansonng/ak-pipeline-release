import matplotlib.pyplot as plt

def get_num_of_regions(side_length, num_of_row_col, coord_lst):
    '''
    Function to get the number of regions having cysts in a grid.
    Parameters:
        side_length (int): the side length of the image (should be 384)
        num_of_row_col (int): the number of rows and columns in the grid
        coord_lst (list): the list of coordinates of cysts, each coordinate is a tuple (x, y). x,y >= 0 & < side_length
    Returns:
        region_count_lst (list): the list of regions, where each region is a number. 0 means no cyst, 1-inf means cyst
    '''

    # get the region size
    region_size = side_length // num_of_row_col
    # the list to store the regions, 0 means no cyst, 1-inf means cyst
    region_count_lst = [0] * (num_of_row_col ** 2)
    # iterate through the coordinates
    for coord in coord_lst:
        # get the region number
        region_num = (coord[0] // region_size) * num_of_row_col + (coord[1] // region_size)
        # set the region to True
        region_count_lst[region_num] += 1
    

    # to get the number of regions having cysts, i.e. having value > 0
    # region_num = len([region for region in region_lst if region > 0])
    return region_count_lst
    


def plot_avg_grid(side_length, num_of_row_col, coord_lst):
    '''
    Function to plot the average grid of cysts
    Parameters:
        side_length (int): the side length of the image (should be 384)
        num_of_row_col (int): the number of rows and columns in the grid
        coord_lst (list): the list of coordinates of cysts, each coordinate is a tuple (x, y). x,y >= 0 & < side_length
    Returns:
        None
    '''

    region_info = get_num_of_regions(side_length, num_of_row_col, coord_lst)

    # reshape the region_info to a matrix
    region_info_reshape = [region_info[i:i+num_of_row_col] for i in range(0, len(region_info), num_of_row_col)]
    print(region_info_reshape)

    # plot image matrix, each cell is the number of cysts in that region
    # region_info[0] should be the number of cysts in the first region (top left corner)
    
    # plot the matrix
    fig, ax = plt.subplots()
    ax.matshow(region_info_reshape, cmap='Greys')
    # color bar
    cbar = plt.colorbar(ax.matshow(region_info_reshape, cmap='Greys'))
    ax.set_title('Average Grid of Cysts')
    plt.show()

if __name__ == "__main__":
    # example usage
    side_length = 384
    num_of_row_col = 2
    coord_lst = [(0, 0), (0, 0), (383, 383)]
    print(get_num_of_regions(side_length, num_of_row_col, coord_lst))
    plot_avg_grid(side_length, num_of_row_col, coord_lst)