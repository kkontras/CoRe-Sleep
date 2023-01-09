def get_attention_weights(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights = io_dict['inner_tf_mod0_l3_RA.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3_RA.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    print(preds.shape)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #

    token_of_interest = 0
    idx = 0

    print(inner_weights.shape)
    print(outer_weights.shape)

    print(target)

    # c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    c = { 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    total_heads = 8
    inner_weights = einops.rearrange(inner_weights,"(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l, h=total_heads)
    outer_weights = einops.rearrange(outer_weights,"(outer h) d m-> outer h d m", outer=batch, h=total_heads)
    print(preds[idx]==target[idx])
    for token_of_interest in [0]:

        plt.figure(figsize=(20, 30))
        x = np.linspace(0, 29, 30).astype(int)
        colors = ["lightblue" for i in x]
        # colors[0] = "lightgreen"
        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Inner Seq Steps")
            plt.xticks(x)
            plt.yticks(x)
            plt.title("CLS Token head {}".format(h))
            plt.imshow(inner_weights[idx][token_of_interest][h], cmap='hot', interpolation='nearest')
        im_ratio = inner_weights[idx][token_of_interest][h].shape[0] / inner_weights[idx][token_of_interest][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)

            # plt.bar(x, inner_weights[idx][token_of_interest][h][0], color=colors)
        plt.show()
        plt.figure(figsize=(25, 20))

        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Outer Seq Steps")
            plt.title("Outer single modality EEG head {}".format(h))
            x = np.linspace(0,len(outer_weights[idx][h][token_of_interest])-1,len(outer_weights[idx][h][token_of_interest])).astype(int)
            l = {0:"W",1:"N1",2:"N2",3:"N3",4:"R"}
            # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
            labels = [l[target[idx][i]] for i in range(len(target[idx]))]
            colors = preds[idx]== target[idx]
            colors = ["lightblue" if i else "red" for i in colors]
            # colors[token_of_interest] = "lightgreen"
            plt.xticks(x,labels, rotation=0)
            plt.yticks(x,labels, rotation=0)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
                ticklabel.set_color(tickcolor)
            for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
                ticklabel.set_color(tickcolor)

            plt.xlabel("Prediction / Correct Label")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
            plt.legend(handles, list(c.keys()))
            plt.imshow(outer_weights[idx][h], cmap='hot', interpolation='nearest')
            im_ratio = outer_weights[idx][h].shape[0] / outer_weights[idx][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)
            # plt.bar(x, outer_weights[idx][h][token_of_interest], color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_merged(model, batch, seq_l, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(target)

    inner_weights = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.subplot(211)
    # plt.ylabel("Attention Weight EEG")
    # plt.title("CLS Token")
    # x = np.linspace(0,59,60).astype(int)
    # colors = ["lightblue" for i in range(len(x))]
    # colors[0] = "lightgreen"
    # # print(inner_weights[0][0].shape)
    # # print(x.shape)
    # plt.bar(x, inner_weights[0][0], color=colors)
    #
    # plt.subplot(212)
    # plt.ylabel("Attention Weight EOG")
    # plt.xlabel("Inner Seq Steps")
    # colors = ["lightblue" for i in range(len(x))]
    # colors[-1] = "lightgreen"
    # plt.bar(x, inner_weights[0][-1], color=colors)
    # plt.show()
    token_of_interest = 0
    idx = -1

    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(ids[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,int(len(outer_weights[idx][token_of_interest])/2)-1, int(len(outer_weights[idx][token_of_interest])/2) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]
        print(len(labels))
        print(x.shape)
        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors_eeg = copy.deepcopy(colors)
        if token_of_interest < len(outer_weights[idx][token_of_interest])/2:
            colors_eeg[token_of_interest] = "lightgreen"
        else:
            colors[token_of_interest - int(len(colors_eeg))] = "lightgreen"
        plt.xticks(x,labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        plt.bar(x, outer_weights[idx][token_of_interest][:int(len(outer_weights[idx][token_of_interest])/2)], color=colors_eeg, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        plt.bar(x, outer_weights[idx][token_of_interest][int(len(outer_weights[idx][token_of_interest]) / 2):],
                color=colors, width=0.7)

        plt.show()

    c = { 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    total_heads = 1
    inner_weights = einops.rearrange(inner_weights,"(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l*2, h=total_heads)
    outer_weights = einops.rearrange(outer_weights,"(outer h) d m-> outer h d m", outer=batch, h=total_heads)
    print(preds[idx]==target[idx])
    for token_of_interest in [5]:

        plt.figure(figsize=(20, 30))
        x = np.linspace(0, inner_weights.shape[1]-1,  inner_weights.shape[1]).astype(int)
        colors = ["lightblue" for i in x]
        # colors[0] = "lightgreen"
        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Inner Seq Steps")
            plt.xticks(x)
            plt.yticks(x)
            plt.title("CLS Token head {}".format(h))
            plt.imshow(inner_weights[idx][token_of_interest][h], cmap='hot', interpolation='nearest')
        im_ratio = inner_weights[idx][token_of_interest][h].shape[0] / inner_weights[idx][token_of_interest][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)

            # plt.bar(x, inner_weights[idx][token_of_interest][h][0], color=colors)
        plt.show()
        plt.figure(figsize=(25, 20))

        for h in range(total_heads):
            if total_heads>1:
                plt.subplot(int("{}{}{}".format(int(total_heads/2),2,h+1)))
            plt.ylabel("Attention Weight")
            plt.xlabel("Outer Seq Steps")
            plt.title("Outer single modality EEG head {}".format(h))
            x = np.linspace(0,len(outer_weights[idx][h][token_of_interest])-1,len(outer_weights[idx][h][token_of_interest])).astype(int)
            l = {0:"W",1:"N1",2:"N2",3:"N3",4:"R"}
            # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
            labels = ["EEG "+ l[target[idx][i%21]] if i<21 else "EOG "+ l[target[idx][i%21]] for i in range(2*len(target[idx]))]
            colors = preds[idx]== target[idx]
            colors_eeg = ["lightblue" if i else "red" for i in colors]
            colors_eog = ["lightblue" if i else "red" for i in colors]
            colors = colors_eeg + colors_eog
            colors[token_of_interest] = "lightgreen"
            plt.xticks(x,labels, rotation=90)
            plt.yticks(x,labels, rotation=0)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
                ticklabel.set_color(tickcolor)
            for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
                ticklabel.set_color(tickcolor)

            plt.xlabel("Label")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
            plt.legend(handles, list(c.keys()))
            plt.imshow(outer_weights[idx][h], cmap='hot', interpolation='nearest')
            im_ratio = outer_weights[idx][h].shape[0] / outer_weights[idx][h].shape[1]
        plt.colorbar( fraction=0.046 * im_ratio, pad=0.04)
            # plt.bar(x, outer_weights[idx][h][token_of_interest], color=colors, width=0.7)

        plt.show()
    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]
    print(target)
    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_concat(model, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)

    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=41).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()

    print(target.shape)
    print(outer_weights.shape)

    # plt.figure()
    # plt.subplot(211)
    # plt.ylabel("Attention Weight EEG")
    # plt.title("CLS Token")
    # x = np.linspace(0,59,60).astype(int)
    # colors = ["lightblue" for i in range(len(x))]
    # colors[0] = "lightgreen"
    # plt.bar(x, inner_weights[0][0], color=colors)
    #
    # plt.subplot(212)
    # plt.ylabel("Attention Weight EOG")
    # plt.xlabel("Inner Seq Steps")
    # colors = ["lightblue" for i in range(len(x))]
    # colors[-1] = "lightgreen"
    # plt.bar(x, inner_weights[0][-1], color=colors)
    # plt.show()
    token_of_interest = 0
    idx = 15

    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(ids[idx])
    token_of_interests = [0 ]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(20, 10))
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG-EOG Concat Tokens")
        x = np.linspace(0,int(len(outer_weights[idx][token_of_interest]))-1, int(len(outer_weights[idx][token_of_interest])) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]
        print(len(labels))
        print(x.shape)
        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors_eeg = copy.deepcopy(colors)
        if token_of_interest < len(outer_weights[idx][token_of_interest]):
            colors_eeg[token_of_interest] = "lightgreen"
        else:
            colors[token_of_interest - int(len(colors_eeg))] = "lightgreen"
        plt.xticks(x,labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))

        plt.bar(x, outer_weights[idx][token_of_interest], color=colors_eeg, width=0.7)
        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_bottleneck(model, device, data_loader, description, context_points=1):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights_eeg = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    inner_weights_eog = io_dict['inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eeg = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eog = io_dict['outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)
    inner_shape = int(len(preds)/32)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=inner_shape).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #
    colors = ["lightblue" if i else "red" for i in range(len(inner_weights_eeg[0][0]))]
    colors[0] = "lightgreen"
    for i in range(len(colors)-context_points, len(colors)):
        colors[i] = "orange"
    c = {'Token of Interest': 'lightgreen', "Intermediate Steps": "lightblue", "Context":"orange"}

    plt.figure()
    plt.subplot(211)
    plt.ylabel("Attention Weight")
    plt.title("CLS Token in EEG")
    x = np.linspace(0,len(inner_weights_eeg[0][0])-1,len(inner_weights_eeg[0][0])).astype(int)
    plt.xticks([],[])
    plt.ylim([0, 0.13])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
    plt.legend(handles, list(c.keys()))
    plt.bar(x, inner_weights_eeg[0][0], color=colors)
    plt.subplot(212)
    plt.ylim([0, 0.13])
    plt.ylabel("Attention Weight")
    plt.xlabel("Inner Seq Steps")
    plt.title("EOG")
    plt.bar(x, inner_weights_eog[0][0], color=colors)
    plt.show()
    token_of_interest = 0
    idx = 15


    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue", "Context": "orange"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(target)

    print(outer_weights_eeg.shape)
    print(preds[idx]==target[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,len(outer_weights_eeg[idx][token_of_interest])-1, len(outer_weights_eeg[idx][token_of_interest]) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]

        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors[token_of_interest] = "lightgreen"
        for i in range(context_points):
            labels.append("C")
            colors.append("orange")
        plt.xticks(x, labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        print(outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])])

        plt.bar(x, outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])], color=colors, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        print(outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])])
        plt.bar(x, outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])],
                color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_late(model, device, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    forward_hook_manager.add_hook(model, 'outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        views = [data[i].float().to(device) for i in range(len(data))]
        preds = get_predictions_time_series(model, views, init)
        break

    io_dict = forward_hook_manager.pop_io_dict()
    print(io_dict.keys())

    inner_weights_eeg = io_dict['inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    inner_weights_eog = io_dict['inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eeg = io_dict['outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    outer_weights_eog = io_dict['outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    target = target.detach().cpu().numpy()

    preds = preds.argmax(dim=1)
    inner_shape = int(len(preds)/32)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=32, b=inner_shape).detach().cpu().numpy()

    t = np.arange(0, 29)
    f = (np.arange(0, 128) / 128) * 50
    this_data = data[0][0, 0, 0, 1:, :].detach().numpy()

    # plt.figure()
    # plt.title("The datapoint we examine")
    # plt.xlabel("Time bins (sec)")
    # plt.ylabel("Freq bins (Hz)")
    # plt.pcolormesh(t, f, this_data, vmin=this_data.min(), vmax=this_data.max(), shading='gouraud')
    # plt.colorbar()
    # plt.show()
    #
    colors = ["lightblue" if i else "red" for i in range(len(inner_weights_eeg[0][0]))]
    colors[0] = "lightgreen"

    c = {'Token of Interest': 'lightgreen', "Intermediate Steps": "lightblue"}

    plt.figure()
    plt.subplot(211)
    plt.ylabel("Attention Weight")
    plt.title("CLS Token in EEG")
    x = np.linspace(0,len(inner_weights_eeg[0][0])-1,len(inner_weights_eeg[0][0])).astype(int)
    plt.xticks([],[])
    plt.ylim([0, 0.13])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
    plt.legend(handles, list(c.keys()))
    plt.bar(x, inner_weights_eeg[0][0], color=colors)
    plt.subplot(212)
    plt.ylim([0, 0.13])
    plt.ylabel("Attention Weight")
    plt.xlabel("Inner Seq Steps")
    plt.title("EOG")
    plt.bar(x, inner_weights_eog[0][0], color=colors)
    plt.show()
    token_of_interest = 0
    idx = -5


    c = {'Token of Interest': 'lightgreen', 'Wrongly classified': 'red', "Correct classified": "lightblue"}
    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    print(target)

    print(outer_weights_eeg.shape)
    print(preds[idx]==target[idx])
    token_of_interests = [0]
    for token_of_interest in token_of_interests:
        plt.figure(figsize=(25, 20))
        plt.subplot(211)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EEG Tokens")
        x = np.linspace(0,len(outer_weights_eeg[idx][token_of_interest])-1, len(outer_weights_eeg[idx][token_of_interest]) ).astype(int)

        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        labels = [l[target[idx][i]] for i in range(len(target[idx]))]

        colors = preds[idx]== target[idx]
        colors = ["lightblue" if i else "red" for i in colors]
        colors[token_of_interest] = "lightgreen"
        plt.xticks(x, labels, rotation=0)
        plt.ylim([0,0.1])
        plt.xlabel("Correct Label")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c[l]) for l in list(c.keys())]
        plt.legend(handles, list(c.keys()))
        print(outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])])

        plt.bar(x, outer_weights_eeg[idx][token_of_interest][:len(outer_weights_eeg[idx][token_of_interest])], color=colors, width=0.7)

        plt.subplot(212)
        plt.ylabel("Attention Weight")
        plt.xlabel("Outer Seq Steps")
        plt.title("EOG Tokens")
        # labels = [l[preds[idx][i]]+"/"+l[target[idx][i]] for i in range(len(target[idx]))]
        plt.ylim([0,0.1])
        plt.xticks(x, labels, rotation=0)
        plt.xlabel("Correct Label")
        print(outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])])
        plt.bar(x, outer_weights_eog[idx][token_of_interest][:len(outer_weights_eog[idx][token_of_interest])],
                color=colors, width=0.7)

        plt.show()

    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,29,30).astype(int)
    # for i in range(1,30):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Inner Seq Steps")
    #     ax.set_title("Inner Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, inner_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('inner.mp4', writer=writer)
    # ani.save('inner.gif', 'imagemagick')


    # figure_list = []
    # fig, ax = plt.subplots()
    # x = np.linspace(0,20,21).astype(int)
    # for i in range(21):
    #     ax.set_ylabel("Attention Weight")
    #     ax.set_xlabel("Outer Seq Steps")
    #     ax.set_title("Tokens in an untrained model")
    #     figure_list.append(ax.bar(x, outer_weights[0][i].detach().numpy(), color='lightblue'))
    # ani = animation.ArtistAnimation(fig, figure_list, blit=False)
    # writer = animation.FFMpegWriter(fps=3, extra_args=['-vcodec', 'libx264'])
    # ani.save('outer.mp4', writer=writer)
    # ani.save('outer.gif', 'imagemagick')
    preds = preds[idx]
    target = target[idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0
def get_attention_weights_late_contrastive(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"

    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i), requires_input=False, requires_output=True)

        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention', requires_input=True, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l0.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l1.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l2.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l3.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    #
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l0.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l1.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l2.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1_l3.outer_tf.layers.0.norm_calc', requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'layer1.0.bn2', requires_input=True, requires_output=True)
    # forward_hook_manager.add_hook(model, 'fc', requires_input=False, requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()

    outer_norms_eeg = {"norm":{}}
    outer_norms_eog = {"norm":{}}
    inner_norms_eeg = {"norm":{}}
    inner_norms_eog = {"norm":{}}
    for i in range(num_layers):
        outer_norms_eeg["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms_eeg["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i)]['output']
        outer_norms_eog["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms_eog["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod1.inner_tf.layers.{}.norm_calc'.format(i)]['output']

    inner_weights_eeg = {"att":{}}
    inner_weights_eog = {"att":{}}
    outer_weights_eeg = {"att":{}}
    outer_weights_eog = {"att":{}}

    for i in range(num_layers):
        inner_weights_eeg["att"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        inner_weights_eog["att"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        outer_weights_eeg["att"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']
        outer_weights_eog["att"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod1.outer_tf.layers.{}.self_attn_my.scaled_dotproduct_attention'.format(i)]['output']

    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_weights_eeg = {}
    # outer_weights_eeg["layer_0"] = io_dict['enc_0.outer_tf_mod0_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_1"] = io_dict['enc_0.outer_tf_mod0_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_2"] = io_dict['enc_0.outer_tf_mod0_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eeg["layer_3"] = io_dict['enc_0.outer_tf_mod0_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_weights_eog = {}
    # outer_weights_eog["layer_0"] = io_dict['enc_0.outer_tf_mod1_l0.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_1"] = io_dict['enc_0.outer_tf_mod1_l1.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_2"] = io_dict['enc_0.outer_tf_mod1_l2.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # outer_weights_eog["layer_3"] = io_dict['enc_0.outer_tf_mod1_l3.outer_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # outer_norms_eeg = {}
    # outer_norms_eeg["layer_0"] = io_dict['enc_0.outer_tf_mod0_l0.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_1"] = io_dict['enc_0.outer_tf_mod0_l1.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_2"] = io_dict['enc_0.outer_tf_mod0_l2.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eeg["layer_3"] = io_dict['enc_0.outer_tf_mod0_l3.outer_tf.layers.0.norm_calc']['output']
    #
    # outer_norms_eog = {}
    # outer_norms_eog["layer_0"] = io_dict['enc_0.outer_tf_mod1_l0.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_1"] = io_dict['enc_0.outer_tf_mod1_l1.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_2"] = io_dict['enc_0.outer_tf_mod1_l2.outer_tf.layers.0.norm_calc']['output']
    # outer_norms_eog["layer_3"] = io_dict['enc_0.outer_tf_mod1_l3.outer_tf.layers.0.norm_calc']['output']

    target = target.detach().cpu().numpy()
    preds = preds.argmax(dim=1)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    print(target)

    total_layers = 4
    total_heads = 8
    batch_idx = -1
    token_of_interest = -4

    l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    labels = [l[target[batch_idx][i % 21]] for i in range(len(target[batch_idx]))]

    neigh_rest_ratio_per_layer = []
    diag_rest_ratio_per_layer = []
    for layer in range(4):
        head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = outer_norms_eeg["norm"]["layer_{}".format(layer)]

        neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
        neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
        neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
        neigh_diag_mask += neigh_diag_2_mask

        diag_mask = torch.eye(attnresln_n.shape[2])
        rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
        neigh_diag_mask = neigh_diag_mask > 0

        attnresln_n_diag = attnresln_n[:,diag_mask>0]
        attnresln_n_rest = attnresln_n[:,diag_mask<1]

        attnresln_n_neigh_rest = attnresln_n[:, rest_mask]
        attnresln_n_neigh = attnresln_n[:, neigh_diag_mask]

        neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
        diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

        neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
        diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

        print("Our neighboring ratio is {}".format(neigh_rest_ratio))
        print("Our diag ratio is {}".format(diag_rest_ratio))
        plt.figure()
        df = pd.DataFrame(attnresln_n[batch_idx].detach().cpu().numpy(), columns=labels , index=labels)
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
        plt.title("Context Ratio layer {}".format(layer))
        plt.show()

    t = np.concatenate(
        [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
        np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
        axis=0)
    plt.figure()
    df = pd.DataFrame(t , columns=["Layer 0","Layer 1","Layer 2","Layer 3"] , index=["Neighbor R","Context R"])
    sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
    plt.title("Ratios Per Layer")
    plt.show()

    #   hidden_states: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
    #   attention_probs: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)
    #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
    #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
    #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

    weights = [inner_weights_eeg["att"], inner_weights_eog["att"], outer_weights_eeg["att"], outer_weights_eog["att"]]
    labels = ["Inner Weights EEG", "Inner Weights EOG", "Outer Weights EEG", "Outer Weights EOG"]

    for plot_num in range(len(weights)):
        fig, ax = plt.subplots()
        plt.title(labels[plot_num])
        plt.ylabel("Layers")
        plt.xlabel("Heads")
        plt.box(on=None)
        plt.xticks([])
        plt.yticks([])
        for l in range(total_layers):
            w = weights[plot_num]["layer_{}".format(l)][1]
            if plot_num<2:
                w = einops.rearrange(w, "(outer inner h) d m-> outer inner h d m", outer=batch, inner=seq_l, h=total_heads)
                w = w[batch_idx][token_of_interest].detach().cpu().numpy()
            else:
                w = einops.rearrange(w, "(outer h) d m-> outer h d m", outer=batch, h=total_heads)
                w = w[batch_idx].detach().cpu().numpy()
            for h in range(total_heads):
                current_subplot = (l * total_heads) + h + 1
                ax = fig.add_subplot(total_layers, total_heads, current_subplot)
                ax.axis('off')
                ax.imshow(w[h], cmap='OrRd_r', interpolation='nearest')
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.show()

    preds = preds[batch_idx]
    target = target[batch_idx]

    non_matches = (preds != target).astype(int)
    non_matches_idx = non_matches.nonzero()[0]
    print("Non matching indices are:")
    hours = len(target)

    non_matches_idx = non_matches_idx

    pred_plus = copy.deepcopy(preds)
    pred_plus[pred_plus == 4] = 5
    pred_plus[pred_plus == 3] = 4
    pred_plus[pred_plus == 2] = 3
    pred_plus[pred_plus == 5] = 2

    target_plus = copy.deepcopy(target)
    target_plus[target_plus == 4] = 5
    target_plus[target_plus == 3] = 4
    target_plus[target_plus == 2] = 3
    target_plus[target_plus == 5] = 2

    # target = target + 0.02
    target_plus = target_plus + 0.02

    plt.figure()
    plt.plot(pred_plus,label="Prediction")
    plt.plot(target_plus,label="True Label")
    plt.scatter(non_matches_idx, pred_plus[non_matches_idx], marker='*', edgecolors="r", label="Mistakes")
    # plt.plot(non_matches_idx,"*")
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "REM", "N2", "N3"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}".format(i) for i in range((hours // 120) + 1)])
    plt.legend()
    plt.ylabel("Labels")
    plt.xlabel("Hours")
    plt.show()

    return 0

def get_attention_weights_late_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()
    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0.outer_tf.layers.{}.norm_calc'.format(i)]['output']
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0.inner_tf.layers.{}.norm_calc'.format(i)]['output']

    target = target.detach().cpu().numpy()
    for weights in [outer_norms]:

        idx = -5
        l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        labels = [l[target[idx][i % 21]] for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(layer)]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_3_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=3) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=-3)
            neigh_diag_mask += neigh_diag_2_mask +neigh_diag_3_mask

            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            norms_matrix = attnresln_n

            attnresln_n_diag = norms_matrix[:,diag_mask>0]
            attnresln_n_rest = norms_matrix[:,diag_mask<1]

            attnresln_n_neigh_rest = norms_matrix[:, rest_mask]
            attnresln_n_neigh = norms_matrix[:, neigh_diag_mask]

            neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}".format(diag_rest_ratio))
            plt.figure()
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            df = pd.DataFrame(norms_matrix[idx].detach().cpu().numpy())
            sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["Neighbor R","Context R"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()

    return 0

    for l_i in range(4):
        plt.figure()
        summed_afx_norm = weights["norm"][l_i][3]
        print(summed_afx_norm.shape)
        norm = summed_afx_norm.detach().cpu()
        diag_mask = torch.eye(norm.shape[2])
        # norm[:, diag_mask > 0] = 0
        norm = norm[idx].numpy()
        print(norm.shape)
        # df = pd.DataFrame(norm )
        df = pd.DataFrame(norm, columns=labels, index=labels )
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize":4})
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.title("neighboring ratio layer {}".format(l_i))
        plt.show()

    return 0

    print(attnresln_n_ratio)

    fig = plt.figure()
    grid = plt.GridSpec(2, 6)
    ratio_inner = torch.cat([inner_weights_eeg["norm"][i][-1][0].unsqueeze(dim=0) for i in range(4)],dim=0).cpu().numpy()
    ratio_outer = torch.cat([outer_weights_eeg["norm"][i][-1][0].unsqueeze(dim=0) for i in range(4)],dim=0).cpu().numpy()
    df_inner = pd.DataFrame(ratio_inner)
    df_outer = pd.DataFrame(ratio_outer)
    plt.subplot(grid[0,0:5])
    sns.heatmap(df_inner, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xticks([])
    plt.title("Inner Ratio")
    plt.ylabel("Layers")

    fig.axes[1].set_visible(False)
    plt.subplot(grid[1,0:5])
    plt.title("Outer Ratio")
    sns.heatmap(df_outer, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xlabel("Sequence")
    plt.ylabel("Layers")
    plt.show()

    fig = plt.figure()
    grid = plt.GridSpec(4, 12)
    for layer in range(4):
        plt.subplot(grid[layer, 0:5])
        summed_afx_norm = inner_weights_eeg["norm"][layer][3]
        norm = summed_afx_norm[0].cpu().numpy()
        df = pd.DataFrame(norm)
        sns.heatmap(df, cmap="Reds", square=True, cbar=False)
        plt.gcf().subplots_adjust(bottom=0.2)
        # fig.axes[layer*2+1].set_visible(False)

        plt.subplot(grid[layer, 6:11])
        summed_afx_norm = inner_weights_eeg["norm"][layer][3]
        norm = summed_afx_norm[0].cpu().numpy()
        df = pd.DataFrame(norm)
        sns.heatmap(df, cmap="Reds", square=True, cbar=False)
        plt.gcf().subplots_adjust(bottom=0.2)
        # fig.axes[(layer+1)*2+1].set_visible(False)


        # plt.title("AttnResLn-N visualization Layer {}".format(layer+1))
    plt.show()


    # Set the layer and head you want to check. (layer: 1~12, head: 1~12)
    layer = 4
    head = 8
    target = target.detach().cpu().numpy()

    plt.figure()
    attention = weights["attention"][layer - 1][0][head - 1].detach().cpu().numpy()
    df = pd.DataFrame(attention)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-W visualization head")
    plt.show()

    plt.figure()
    afx_norm = weights["norm"][layer - 1][0]
    norm = afx_norm[0][head - 1].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-N visualization head")
    plt.show()

    plt.figure()
    attention = weights["attention"][layer - 1][0].mean(0).detach().cpu().numpy()
    df = pd.DataFrame(attention)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-W visualization layer")
    plt.show()

    plt.figure()
    summed_afx_norm = weights["norm"][layer - 1][1]
    norm = summed_afx_norm[0].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Attn-N visualization layer")
    plt.show()

    plt.figure()
    attention = weights["attention"][layer - 1][0].mean(0).detach().cpu().numpy()
    res = np.zeros((len(attention), len(attention)), int)
    np.fill_diagonal(res, 1)
    attnres_w = 0.5 * attention + 0.5 * res
    df = pd.DataFrame(attnres_w)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("AttnRes-W visualization layer")
    plt.show()

    plt.figure()
    summed_afx_norm = weights["norm"][layer - 1][2]
    norm = summed_afx_norm[0].cpu().numpy()
    df = pd.DataFrame(norm)
    sns.heatmap(df, cmap="Reds", square=True)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("AttnRes-N visualization")
    plt.show()


    return 0

    target = target.detach().cpu().numpy()
    preds = preds.argmax(dim=1)
    preds = einops.rearrange(preds,"(a b) -> a b ", a=batch, b=seq_l).detach().cpu().numpy()

    print(target)

    total_layers = 4
    total_heads = 8
    batch_idx = -4
    token_of_interest = 0

    print(outer_weights_eeg["af"]["layer_0"].shape)
    print(outer_weights_eeg["f"]["layer_0"].shape)
    print(outer_weights_eeg["attention"]["layer_0"].shape)

    # weights = [outer_weights_eeg["af"], outer_weights_eeg["f"], outer_weights_eeg, outer_weights_eog]
    # labels = [ "Outer Weights EEG"]
    fig, ax = plt.subplots()
    plt.title("Function Norm Analysis")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        print(w.shape)
        w = einops.rearrange(w, "(batch h) a b-> batch h (a b)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 1)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|Att|")

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["f"]["layer_{}".format(l)]
        w = einops.rearrange(w, "outer (batch h) f-> batch h (outer f)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 2)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|F(x)|")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    total_image = []
    for l in range(total_layers):
        w = outer_weights_eeg["af"]["layer_{}".format(l)]
        w = einops.rearrange(w, "outer (batch h) f-> batch h (outer f)", batch=batch,  h=total_heads)
        w = np.linalg.norm(w[batch_idx], axis=-1)
        total_image.append(np.expand_dims(w,axis=0))
    total_image = numpy.concatenate(total_image, axis=0)

    ax = fig.add_subplot(1, 3, 3)
    # ax.axis('off')
    ax.imshow(total_image, cmap='OrRd_r', interpolation='nearest')
    ax.set_xlabel("Heads")
    ax.set_ylabel("Layers")
    ax.set_title("|α F(x)|")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.show()


    fig, ax = plt.subplots()
    plt.title("Attention weights")
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        w = einops.rearrange(w, "(outer h) d m-> outer h d m", outer=batch, h=total_heads)
        w = w[batch_idx]
        for h in range(total_heads):
            current_subplot = (l * total_heads) + h + 1
            ax = fig.add_subplot(total_layers, total_heads, current_subplot)
            ax.axis('off')
            ax.imshow(w[h], cmap='OrRd_r', interpolation='nearest')
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Attention weights normed with F")
    plt.ylabel("Layers")
    plt.xlabel("Heads")
    plt.box(on=None)
    plt.xticks([])
    plt.yticks([])
    for l in range(total_layers):
        w = outer_weights_eeg["attention"]["layer_{}".format(l)]
        f = outer_weights_eeg["f"]["layer_{}".format(l)]
        f = einops.rearrange(f, "outer (batch h) f-> batch outer (h f)", batch=batch,  h=total_heads)
        f = np.linalg.norm(f[batch_idx], axis=-1)
        print(f.shape)
        w = einops.rearrange(w, "(b h) d m-> b h d m", b=batch, h=total_heads)
        w = w[batch_idx]

        for i in range(len(f)):
            w[:,:,i] *= f[i]

        from scipy.special import softmax

        for h in range(total_heads):
            current_subplot = (l * total_heads) + h + 1
            ax = fig.add_subplot(total_layers, total_heads, current_subplot)
            ax.axis('off')
            ax.imshow( softmax(w[h],axis=-1), cmap='OrRd_r', interpolation='nearest')
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

    return 0
def get_attention_weights_late_retarded_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod0_l{}.outer_tf.layers.0.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod0_l{}.inner_tf.layers.0.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:

        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf_mod0_l{}.outer_tf.layers.0.norm_calc'.format(i)]['output']
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf_mod0_l{}.inner_tf.layers.0.norm_calc'.format(i)]['output']

    target = target.detach().cpu().numpy()
    for weights in [outer_norms]:

        idx = -5
        l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        labels = [l[target[idx][i % 21]] for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(layer)]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_3_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=3) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 3, dtype=torch.long), offset=-3)
            neigh_diag_mask += neigh_diag_2_mask +neigh_diag_3_mask

            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            norms_matrix = attnresln_n

            attnresln_n_diag = norms_matrix[:,diag_mask>0]
            attnresln_n_rest = norms_matrix[:,diag_mask<1]

            attnresln_n_neigh_rest = norms_matrix[:, rest_mask]
            attnresln_n_neigh = norms_matrix[:, neigh_diag_mask]

            neigh_rest_ratio = attnresln_n_neigh.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}/{}+{} = {}".format(attnresln_n_rest.mean(),attnresln_n_rest.mean(), attnresln_n_diag.mean(), diag_rest_ratio))
            plt.figure()
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            df = pd.DataFrame(norms_matrix[idx].detach().cpu().numpy())
            sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["Neighbor R","Context R"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()

    return 0
def get_attention_weights_merged_norm(model, device, batch, seq_l, data_loader, description):

    # device = "cuda:{}".format(config.gpu_device[0])
    # device = "cpu"
    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'enc_0.inner_tf', requires_output=True)
    forward_hook_manager.add_hook(model, 'enc_0.outer_tf', requires_output=True)
    num_layers = 4
    for i in range(num_layers):
        forward_hook_manager.add_hook(model, 'enc_0.outer_tf.outer_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)
        forward_hook_manager.add_hook(model, 'enc_0.inner_tf.inner_tf.layers.{}.norm_calc'.format(i), requires_input=False, requires_output=True)

    # forward_hook_manager.add_hook(model, 'enc_0.inner_tf_mod1', requires_output=True)
    # forward_hook_manager.add_hook(model, 'enc_0.outer_tf_mod1', requires_output=True)

    model.eval()
    pbar = tqdm(enumerate(data_loader), desc=description, leave=False)
    for batch_idx, (data, target, init, ids) in pbar:
        print(batch_idx)
        views = [data[i].float().to(device) for i in range(len(data))]
        # views[0] = torch.cat([views[0][:16],views[0][16:]],dim=1)
        # views[1] = torch.cat([views[1][:16],views[1][16:]],dim=1)
        preds = get_predictions_time_series(model, views, init, extract_norm=True)
        break

    # batch, seq_l = 16, 42

    io_dict = forward_hook_manager.pop_io_dict()
    # print(io_dict.keys())
    # inner_weights_eeg = {}
    # inner_weights_eeg["layer_0"] = io_dict['enc_0.inner_tf_mod0_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_1"] = io_dict['enc_0.inner_tf_mod0_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_2"] = io_dict['enc_0.inner_tf_mod0_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eeg["layer_3"] = io_dict['enc_0.inner_tf_mod0_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    #
    # inner_weights_eog = {}
    # inner_weights_eog["layer_0"] = io_dict['enc_0.inner_tf_mod1_l0.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_1"] = io_dict['enc_0.inner_tf_mod1_l1.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_2"] = io_dict['enc_0.inner_tf_mod1_l2.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()
    # inner_weights_eog["layer_3"] = io_dict['enc_0.inner_tf_mod1_l3.inner_tf.layers.0.self_attn_my.scaled_dotproduct_attention']['output'][1].detach().cpu().numpy()

    outer_norms = {"norm":{}}
    inner_norms = {"norm":{}}
    for i in range(num_layers):
        outer_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.outer_tf.outer_tf.layers.{}.norm_calc'.format(i)]["output"]
        inner_norms["norm"]["layer_{}".format(i)] = io_dict['enc_0.inner_tf.inner_tf.layers.{}.norm_calc'.format(i)]["output"]

    # target = target.detach().cpu().numpy()
    target = target.detach().argmax(dim=-1).cpu().numpy()
    print(target)

    idx = 1
    # colors_b = ["c" for i in range(29)]
    # colors_c = ["y" for i in range(29)]
    # colors = ["k"] + colors_b + colors_c
    # from matplotlib.lines import Line2D
    # for weights in [inner_norms]:
    #     plt.figure()
    #     plt.subplot(221)
    #     i=0
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #     plt.subplot(222)
    #     i=1
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #     plt.subplot(223)
    #     i=2
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #
    #     plt.subplot(224)
    #     i=3
    #     plt.bar(np.arange(59),weights["norm"]["layer_{}".format(i)][1].mean(dim=0)[0].detach().cpu().numpy(), color=colors)
    #     plt.title("CLS layer {}".format(i))
    #     plt.axis("off")
    #
    #     legend_elements = [ Line2D([0], [0], marker='o', color='c', label='EEG',
    #                               markerfacecolor='c', markersize=10),
    #                         Line2D([0], [0], marker='o', color='y', label='EOG',
    #                                markerfacecolor='y', markersize=10),
    #                         Line2D([0], [0], marker='o', color='k', label='CLS',
    #                                markerfacecolor='k', markersize=10)
    #                         ]
    #     plt.legend(handles=legend_elements, loc="lower center")
    #
    #     plt.show()
            # df = pd.DataFrame(weights["norm"]["layer_{}".format(i)][3][idx].detach().cpu().numpy())
            # sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.1g', annot_kws={"fontsize": 4})
            # plt.title("Context Ratio layer {}".format(i))
            # plt.show()

    # print(target[idx])
    for weights in [inner_norms]:

        # l = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        # labels = [l[target[idx][i % 21]] for j in range(2) for i in range(len(target[idx]))]

        neigh_rest_ratio_per_layer = []
        diag_rest_ratio_per_layer = []
        crossmodal_rest_ratio_per_layer_mod0to1 = []
        crossmodal_rest_ratio_per_layer_mod1to0 = []
        run_crossmodal = False
        for layer in range(num_layers):
            head_attn_n, attn_n, attnres_n, attnresln_n, attn_n_ratio, attnres_n_ratio, attnresln_n_ratio = weights["norm"]["layer_{}".format(i)]

            #leave out cls
            attnresln_n = attnresln_n[:,1:,1:]

            neigh_diag = torch.ones(attnresln_n.shape[2] - 1, dtype=torch.long)
            neigh_diag_mask = torch.diagflat(neigh_diag, offset=1) + torch.diagflat(neigh_diag, offset=-1)
            neigh_diag_2_mask = torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=2) + torch.diagflat(torch.ones(attnresln_n.shape[2] - 2, dtype=torch.long), offset=-2)
            neigh_diag_mask += neigh_diag_2_mask

            if run_crossmodal:
                cross_modal_mask_mod0to1 = torch.zeros([attnresln_n.shape[2], attnresln_n.shape[2]])
                cross_modal_mask_mod1to0 = torch.zeros([attnresln_n.shape[2], attnresln_n.shape[2]])
                cross_modal_mask_mod1to0[int(cross_modal_mask_mod1to0.shape[0] / 2):, :int(cross_modal_mask_mod1to0.shape[0] / 2)] = torch.ones([int(attnresln_n.shape[2]/2), int(attnresln_n.shape[2]/2)])
                cross_modal_mask_mod0to1[:int(cross_modal_mask_mod0to1.shape[0] / 2), int(cross_modal_mask_mod0to1.shape[0] / 2):] = torch.ones([int(attnresln_n.shape[2]/2), int(attnresln_n.shape[2]/2)])


            diag_mask = torch.eye(attnresln_n.shape[2])
            rest_mask = (torch.ones([attnresln_n.shape[2], attnresln_n.shape[2]], dtype=torch.long) - diag_mask - neigh_diag_mask) > 0
            neigh_diag_mask = neigh_diag_mask > 0

            attnresln_n_diag = attnresln_n[:,diag_mask>0]
            attnresln_n_rest = attnresln_n[:,diag_mask<1]

            attnresln_n_neigh_rest = attnresln_n[:, rest_mask]
            attnresln_n_neigh = attnresln_n[:, neigh_diag_mask]

            if run_crossmodal:
                attnresln_n_crossmodal_rest_mod0to1 = attnresln_n[:, cross_modal_mask_mod0to1>0]
                attnresln_n_crossmodal_mod0to1 = attnresln_n[:, cross_modal_mask_mod0to1<1]

                attnresln_n_crossmodal_rest_mod1to0  = attnresln_n[:, cross_modal_mask_mod1to0>0]
                attnresln_n_crossmodal_mod1to0  = attnresln_n[:, cross_modal_mask_mod1to0<1]

            neigh_rest_ratio = attnresln_n_neigh_rest.mean() / (attnresln_n_neigh_rest.mean() + attnresln_n_neigh.mean())
            diag_rest_ratio = attnresln_n_rest.mean() / (attnresln_n_rest.mean() + attnresln_n_diag.mean())

            if run_crossmodal:
                crossmodal_rest_ratio_mod0to1 = attnresln_n_crossmodal_rest_mod0to1.mean() / (attnresln_n_crossmodal_rest_mod0to1.mean() + attnresln_n_crossmodal_mod0to1.mean())
                crossmodal_rest_ratio_mod1to0 = attnresln_n_crossmodal_rest_mod1to0.mean() / (attnresln_n_crossmodal_rest_mod1to0.mean() + attnresln_n_crossmodal_mod1to0.mean())

            neigh_rest_ratio_per_layer.append(neigh_rest_ratio.detach().cpu().numpy())
            diag_rest_ratio_per_layer.append(diag_rest_ratio.detach().cpu().numpy())
            if run_crossmodal:
                crossmodal_rest_ratio_per_layer_mod0to1.append(crossmodal_rest_ratio_mod0to1.detach().cpu().numpy())
                crossmodal_rest_ratio_per_layer_mod1to0.append(crossmodal_rest_ratio_mod1to0.detach().cpu().numpy())

            print("Our neighboring ratio is {}".format(neigh_rest_ratio))
            print("Our diag ratio is {}".format(diag_rest_ratio))
            if run_crossmodal:
                print("Our cross modal context ratio from eeg to eog is {}".format(crossmodal_rest_ratio_mod0to1))
                print("Our cross modal context ratio from eog to eeg is {}".format(crossmodal_rest_ratio_mod1to0))

            plt.figure()
            df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy())
            # df = pd.DataFrame(attnresln_n[idx].detach().cpu().numpy(), columns=labels , index=labels)
            sns.heatmap(df, cmap="Blues", square=True, annot=False, fmt='.1g', annot_kws={"fontsize": 4})
            plt.title("Context Ratio layer {}".format(layer))
            plt.show()

        t = np.concatenate(
            [
            # np.expand_dims(np.array(neigh_rest_ratio_per_layer),axis=0),
            # np.expand_dims(np.array(diag_rest_ratio_per_layer),axis=0),
            np.expand_dims(np.array(crossmodal_rest_ratio_per_layer_mod0to1),axis=0),
            np.expand_dims(np.array(crossmodal_rest_ratio_per_layer_mod1to0),axis=0)],
            axis=0)
        plt.figure()
        layer_columns = ["Layer {}".format(i) for i in range(num_layers)]
        df = pd.DataFrame(t , columns=layer_columns , index=["EEG->EOG", "EOG->EEG"])
        sns.heatmap(df, cmap="Blues", square=True, annot=True, fmt='.3g', annot_kws={"fontsize": 8})
        plt.title("Ratios Per Layer")
        plt.show()
