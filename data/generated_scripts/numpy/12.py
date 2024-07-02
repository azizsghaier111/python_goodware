# Masked Array Operations
masked_array = np.ma.masked_array(my_array, mask=[0, 0, 1, 0, 0, 1, 0, 0, 1])
print('Masked Array:', masked_array)

# Accessing and modifying masked values
print("Accessing first masked value:", masked_array[2])
masked_array[2] = 100
print('Masked Array after modification:', masked_array)

# Mathematical operations preserve the mask
print("Masked array addition:", masked_array + 10)

# Basic Mathematical Functions
print(' sine:', np.sin(my_array))
print(' cosine:', np.cos(my_array))
print(' tangent:', np.tan(my_array))
print('Inverse sine:', np.arcsin(my_array/9))
print('Inverse cosine:', np.arccos(my_array/9))
print('Inverse tangent:', np.arctan(my_array/9))
print('Hyperbolic sine:', np.sinh(my_array/9))
print('Hyperbolic cosine:', np.cosh(my_array/9))
print('Hyperbolic tangent:', np.tanh(my_array/9))
print('Inverse hyperbolic sine:', np.arcsinh(my_array/9))
print('Inverse hyperbolic cosine:', np.arccosh(my_array/9))
print('Inverse hyperbolic tangent:', np.arctanh(my_array/9))
print('Rounding array below 0.5 to nearest integer:', np.round([0.3, 0.6, 0.4]))
print('Floor of the array:', np.floor(my_array))
print('Ceil of the array:', np.ceil(my_array))

# Converting Masked Array to a DataFrame
masked_df = pd.DataFrame(masked_array, columns=['Column1'])

# Accessing and modifying DataFrame, including adding new column
masked_df.loc[1, 'Column1'] = 50
masked_df['Column2'] = np.random.randint(10, size=masked_df.shape[0])
print(masked_df)

# Basic Dataframe operations
print(masked_df.describe())
print(masked_df.info())
print(masked_df.head())

# Groupby operation
grouped_df = masked_df.groupby('Column2').sum()
print(grouped_df)

# Concatenating and Merging DataFrames
merged_df = pd.concat([df, masked_df], keys=['df', 'masked_df'])
merged_df.reset_index(level=0, inplace=True)
print(merged_df)

# Saving to a csv file, and loading from a csv file
merged_df.to_csv('merged.csv')
loaded_df = pd.read_csv('merged.csv')
print(loaded_df)