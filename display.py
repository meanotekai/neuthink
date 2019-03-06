from reprint import output

output_lines = output(initial_len=5, interval=0)
def display_comp_status(nbatch, batch_loss, test_loss, test_f1, best_loss, best_f1,learning_rate, test_accuracy=None, best_accuracy=None,train_moving_loss=None):
    global output_lines
    test_f1  = " ".join([str(x[0])+":"+str(x[1])[0:4] for x in test_f1 if str(x[0])!="other"])
    best_f1  = " ".join([str(x[0])+":"+str(x[1])[0:4] for x in best_f1 if str(x[0])!="other"])

    output_lines.warped_obj[0] = "Current:  learning rate: " + str(learning_rate) + "  Train error:" + str(train_moving_loss)
    output_lines.warped_obj[1] = " #batch :" + str(nbatch) + " batch loss " + str(batch_loss)
    if test_f1!="":
       output_lines.warped_obj[2] = " Test error :" + str(test_loss) + "   Test F1:" + str(test_f1)
    if test_accuracy is not None:
        output_lines.warped_obj[2] = " Test error :" + str(test_loss) + "   Test accuracy:" + str(test_accuracy)
    else:
        output_lines.warped_obj[2] = " Test error :" + str(test_loss) 
    output_lines.warped_obj[3] = "Best result so far: "
    if best_f1!="":
        output_lines.warped_obj[4] = " Best error :" + str(best_loss) + "   Best F1:" + str(best_f1)
    if best_accuracy is not None:
        output_lines.warped_obj[4] = " Best error :" + str(best_loss) + "   Best accuracy:" + str(best_accuracy)
    else:
        output_lines.warped_obj[4] = " Best error :" + str(best_loss)