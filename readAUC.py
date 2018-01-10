auc = []
for i in range(0,12,1):
    fh = open('/Users/Hongzhi/Downloads/NewCreditCard/' + 'creditcard_save_auc_' + str(i))
    auc.append(fh.read())
    fh.close()
print auc