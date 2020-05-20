
def gen_sin(freq, amplitude, sr=1000):
    return np.sin(
        (freq * 2 * np.pi * np.linspace(0, sr, sr)) / sr
    ) * amplitude

def plot_wave_composition(defs, hspace=1.0):
    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = [14.0, 10.0]

    waves = [
        gen_sin(freq, amp)
        for freq, amp in defs
    ]

    fig, axs = plt.subplots(nrows=len(defs) + 1)

    for ix, wave in enumerate(waves):
        sb.lineplot(data=wave, ax=axs[ix])
        axs[ix].set_ylabel('{}'.format(defs[ix]))

        if ix != 0:
            axs[ix].set_title('+')

    plt.subplots_adjust(hspace = hspace)

    sb.lineplot(data=sum(waves), ax=axs[len(defs)])
    axs[len(defs)].set_ylabel('sum')
    axs[len(defs)].set_xlabel('time')
    axs[len(defs)].set_title('=')

    plt.rcParams["figure.figsize"] = fig_size

    return waves, sum(waves)