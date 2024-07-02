# Masked array operation
masked_array = np.ma.array([1, 2, 3], mask=[0, 1, 0])
print("Initial Masked Array:", masked_array)
masked_array.mask[2] = 1
print("Modified Masked Array:", masked_array)

# Array sorting
unsorted_array = np.array([2, 1, 4, 3, 5])
sorted_array = np.sort(unsorted_array)
print("Sorted Array:", sorted_array)

# Set operations
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])
print("Union:", np.union1d(arr1, arr2))
print("Intersection:", np.intersect1d(arr1, arr2))
print("Difference:", np.setdiff1d(arr1, arr2))

# Existing script
...