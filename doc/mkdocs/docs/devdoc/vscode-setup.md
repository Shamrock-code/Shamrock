# VSCode developement setup

## Repo setup

We assume in this guide that you have clonned the repository.
In order for this to work you need to have a build directory in the root folder named `build`, otherwise vscode will crap itself because it does not understand build directories not named build ...

In summary we assume the following commands

```bash
git clone --recurse-submodules git@github.com:tdavidcl/Shamrock.git
cd Shamrock
./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp
cd build
source ./activate
shamconfigure
```

## VSCode

Many IDE are available for C++ developement, in this guide we focus on VScode.

Many flavors of VScode are available. Either the spyware version [VScode](https://code.visualstudio.com/) or the cleaned version without microsoft telemetry, ai, whatever and especially trully open source [VScodium](https://vscodium.com/). This guide work on both.

In the `Shamrock` folder run either `code` (how can microsoft reserve such command name btw !!!) or `codium` to start the IDE in the corect folder.

Initially you should see something like this

![VScode blank](../assets/large-figures/figures/vscode/vscode_blank.png)

## VSCode profiles

Start by creating a new vscode profile to avoid messing up existing configurations (you can also import existing keyboard shortcut or whatever at this step, see: [VScode profiles](https://code.visualstudio.com/docs/editor/profiles)).

Click on the setting icon, go in the profile tab and select `Create Profile...`
![VScode blank](../assets/large-figures/figures/vscode/create_profile.png)
Select your options to create the profile and click on `Create`
![VScode blank](../assets/large-figures/figures/vscode/create_profile2.png)

Now that you have created a profile, go to the extension tab,
![VScode blank](../assets/large-figures/figures/vscode/go_to_ext.png)
and install the `clangd` C++ language server to get autocompletion and syntax highlight/checking.
![VScode blank](../assets/large-figures/figures/vscode/install_clangd.png)

Clangd will be looking at the file `build/compile_commands.json` from the root directory to get the compilation arguments. In Shamrock the `.clangd` file in the root of the directory and the CMake configuration is made in such a way that you will get autocompletion with SYCL support in the repository. To check go in any cpp file (`src/main.cpp` here) and it should work (check that is says `clangd: idle` or something similar at the bottom of the screen).
![VScode blank](../assets/large-figures/figures/vscode/it_works.png)

You can now go on the menu bar in the terminal tab and open a new terminal and you can start working normally.
![VScode blank](../assets/large-figures/figures/vscode/compile.png)

You can then install whatever VSCode extensions of your liking, just avoid any intelisense c++ extensions as they conflict with Clangd (which is arguably better btw 😄).
