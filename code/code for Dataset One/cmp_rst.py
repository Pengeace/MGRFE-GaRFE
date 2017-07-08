import matplotlib.pyplot as plt

ga_accs =  [1, 0.981, 0.985, 1, 0.937]
mc_accs =  [1,  0.95,  0.90, 1, 0.85]
ga_feats = [3, 3, 6, 2, 7]
mc_feats = [4, 3, 6, 2, 7]
data_names = ['DLBCL', 'Pros', 'Colon', 'Leuk', 'Mye']

ga_accs +=  [ 1, 0.91, 0.927, 0.99, 1, 1, 1]
mc_accs +=  [ 1, 0.75,  0.80, 0.88, 0.85, 1, 1]
ga_feats += [ 1, 8, 8, 6, 7, 3, 1]
mc_feats += [ 1, 2, 5, 2, 4, 4, 2]
data_names += [ 'ALL1', 'ALL2', 'ALL3', 'ALL4', 'CNS', 'Lym', 'Adeno']

ga_accs  += [    1, 0.98, 1, 0.91,    1]
mc_accs  += [ 0.97, 0.95, 1, 0.81, 0.85]
ga_feats += [ 3, 3, 2, 7, 4]
mc_feats += [ 3, 4, 2, 6, 1]
data_names += ['Gas', 'Gas1', 'Gas2', 'T1D', 'Stroke']

output_path = "D:\\codes\\python\\MachineLearning\\Python_Lesson\\ftselect\\final_version\\cv\\"
plt.figure(figsize=(14, 6), dpi=80)
plt.subplot(1,1,1)
plt.yticks([0.1 * i for i in range(5,11)])
plt.xticks(range(len(data_names)), data_names)
X = range(len(data_names))
plt.ylim(0.5, 1.1)
plt.xlim(-0.5, len(data_names))
plt.ylabel('mAcc', fontsize=15)
plt.plot(X, ga_accs, label='MGRfe', marker='s', linewidth=2.5, markersize=10,color='deepskyblue')
plt.plot(X, mc_accs, label='McTwo', marker='d', linewidth=2, markersize=10,color='lime')
plt.legend(loc='upper right')
plt.savefig(output_path+"cmp_rst.pdf")
plt.show()
plt.close()