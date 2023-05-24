

def plot_observables(model1,
                    model2, 
                    model3, 
                    test_all,  
                    x1_sample, 
                    x2_sample, 
                    n_sample, 
                    xw, 
                    xw_compare,
                    foldername):
    
    energy1 = model1(test_all)
    energy2 = model2(test_all)
    energy3 = model3(test_all)

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    area = 10
    Energy_landscape = (energy1-energy1.min()).detach() 
    levels = np.linspace(0,10,20)
    contour = axes[0].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[0].scatter(xw[:,0].cpu().detach().numpy(), xw[:,1].cpu().detach().numpy(),c = "tab:orange", s=area, alpha=0.5)
    cbar1 = plt.colorbar(contour)
    axes[0].set_title("Energy Function (w/ Jarzynski)")
    plot_limit = 100
    axes[0].set_xlim(-plot_limit,plot_limit)
    axes[0].set_ylim(-plot_limit,plot_limit)

    Energy_landscape = (energy2-energy2.min()).detach() 
    contour = axes[1].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[1].scatter(xw_compare[:,0].cpu().detach().numpy(), xw_compare[:,1].cpu().detach().numpy(),c = "tab:blue", s=area, alpha=0.5)
    cbar2 = plt.colorbar(contour)
    axes[1].set_title("Energy Function (w/o Jarzynski)")
    axes[1].set_xlim(-plot_limit,plot_limit)
    axes[1].set_ylim(-plot_limit,plot_limit)

    Energy_landscape = (energy3-energy3.min()).detach() 
    contour = axes[2].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[2].scatter(data[:,0].cpu().detach().numpy(), data[:,1].cpu().detach().numpy(),c = "tab:blue", s=area, alpha=0.5)
    cbar2 = plt.colorbar(contour)
    axes[2].set_title("True Energy Function")
    axes[2].set_xlim(-plot_limit,plot_limit)
    axes[2].set_ylim(-plot_limit,plot_limit)
    filename = str(t) + "_data.png"
    plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    plt.close()
            
            # fig4 = plt.figure(4);fig4.clf
            # # min1=np.abs(np.min(ce.detach().cpu().numpy()[0:t]))
            # # min2=np.abs(np.min(ce2.detach().cpu().numpy()[0:t]))
            # # shift=np.max((min1,min2))+0.1
            # plt.semilogy(ce.detach().cpu().numpy()[0:t]+add_const.cpu().numpy(),label = "Formula 18")
            # plt.semilogy(ce2.detach().cpu().numpy()[0:t]+add_const.cpu().numpy(),label = "Empirical Loss")
            # plt.legend()
            # plt.title("Loss")
            # plt.xlabel("Adam Iterations")
            # plt.ylabel("Loss")
            # filename = str(t) + "_ce.png"
            # plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
            # plt.close()


def plot_observables_2d(model1, 
                        model2, 
                        model3, 
                        test_all, 
                        x1_sample, 
                        x2_sample, 
                        n_sample, 
                        xw, 
                        xw_compare, 
                        data, 
                        foldername, 
                        KL, 
                        KL2, 
                        ce, 
                        part_func):
    
    energy1 = model1(test_all)
    energy2 = model2(test_all)
    energy3 = model3(test_all)

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    area = 10
    Energy_landscape = (energy1-energy1.min()).detach() 
    levels = np.linspace(0,10,20)
    contour = axes[0].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[0].scatter(xw[:,0].cpu().detach().numpy(), xw[:,1].cpu().detach().numpy(),c = "tab:orange", s=area, alpha=0.5)
    cbar1 = plt.colorbar(contour)
    axes[0].set_title("Energy Function (w/ Jarzynski)")
    plot_limit = 100
    axes[0].set_xlim(-plot_limit,plot_limit)
    axes[0].set_ylim(-plot_limit,plot_limit)

    Energy_landscape = (model2(test_all)-model2(test_all).min()).detach() 
    contour = axes[1].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[1].scatter(xw_compare[:,0].cpu().detach().numpy(), xw_compare[:,1].cpu().detach().numpy(),c = "tab:blue", s=area, alpha=0.5)
    cbar2 = plt.colorbar(contour)
    axes[1].set_title("Energy Function (w/o Jarzynski)")
    axes[1].set_xlim(-plot_limit,plot_limit)
    axes[1].set_ylim(-plot_limit,plot_limit)

    Energy_landscape = (energy3-energy3.min()).detach() 
    contour = axes[2].contourf(x1_sample,x2_sample,Energy_landscape.cpu().numpy().reshape((n_sample,n_sample)),levels = levels)
    axes[2].scatter(data[:,0].cpu().detach().numpy(), data[:,1].cpu().detach().numpy(),c = "tab:blue", s=area, alpha=0.5)
    cbar2 = plt.colorbar(contour)
    axes[2].set_title("True Energy Function")
    axes[2].set_xlim(-plot_limit,plot_limit)
    axes[2].set_ylim(-plot_limit,plot_limit)
    filename = str(t) + "_data.png"
    plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    plt.close()

    fig5 = plt.figure(4);fig5.clf
    plt.semilogy(KL.detach().cpu().numpy()[0:q],label = "W/ Jarzynski - On Grid")
    plt.semilogy(KL2.detach().cpu().numpy()[0:q],label = "W/o Jarzynski - On Grid")
    plt.semilogy(ce.detach().cpu().numpy()[0:q]+add_const.cpu().numpy(),label = "W/ Jarzynski - Using weights")
    #         plt.semilogy(total_norm_all.cpu().numpy()[0:t],label = "Regularization W/ Jarzynski")
    plt.legend()
    plt.title("KL Divergence")
    plt.xlabel("Adam Iterations x 1e3")
    plt.ylabel("KL")
    filename = str(t) + "_KL.png"
    plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    plt.close()

    fig5 = plt.figure(4);fig5.clf
    plt.plot(np.exp(part_func[0:t]))


    #         plt.semilogy(total_norm_all.cpu().numpy()[0:t],label = "Regularization W/ Jarzynski")

    plt.title("Partition function")
    plt.xlabel("Adam Iterations")
    plt.ylabel("Z")
    filename = str(t) + "_part.png"
    plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    plt.close()

    # fig4 = plt.figure(4);fig4.clf
    # # min1=np.abs(np.min(ce.detach().cpu().numpy()[0:t]))
    # # min2=np.abs(np.min(ce2.detach().cpu().numpy()[0:t]))
    # # shift=np.max((min1,min2))+0.1
    # plt.semilogy(ce.detach().cpu().numpy()[0:t]+add_const.cpu().numpy(),label = "Formula 18")
    # plt.semilogy(ce2.detach().cpu().numpy()[0:t],label = "Empirical Loss")
    # plt.legend()
    # plt.title("Loss")
    # plt.xlabel("Adam Iterations")
    # plt.ylabel("Loss")
    # filename = str(t) + "_ce.png"
    # plt.savefig(foldername + filename, dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    # plt.close()