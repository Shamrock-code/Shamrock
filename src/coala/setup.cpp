//-------------------------------------------
//setup file to choose parameters
//-------------------------------------------
#include "options.hpp"
#include <string.h>
#include <iostream>
#include <stdbool.h> 

//-------------------------------------------
//choose numerical scheme
//scheme = 0 : kernel approx
//-------------------------------------------
unsigned int scheme = 0;

//-------------------------------------------
//choose kernel
//0->kconst
//1->kadd
//2->kmul
//3->k_ballistic
//-------------------------------------------
unsigned int kernel=1;

//-------------------------------------------
//Display all processes
//-------------------------------------------
bool process = true;

//-------------------------------------------
//Display gij values
//-------------------------------------------
bool results = true;

//-------------------------------------------
//Save gij and gxmeanlog values in data/ directory
//-------------------------------------------
bool save = false;
   





//-------------------------------------------
//function to compute variable for path and create data. directory
//-------------------------------------------
void init_path_files(unsigned int nbins, unsigned int kflux, unsigned int scheme, unsigned int kernel, char path_data[]){

    char strkernel[15];
    char strscheme[15];

    switch (kernel){
        case 0:
            strcpy(strkernel,"kconst");
            break;
        case 1:
            strcpy(strkernel,"kadd");
            break;
        case 2:
            strcpy(strkernel,"kmul");
            break;
        case 3:
            strcpy(strkernel,"k_ballistic");
            break;
        default:
            printf("issue in choosing kernel, setup \n");
            // exit(-1);
    }

    switch (scheme) {
        case 0:
            strcpy(strscheme,"kernel_approx");
            break;
        default:
            printf("issue in choosing scheme, setup \n");
            // exit(-1);
    }

    char strnbins[3];
    char strkflux[2]; 
    sprintf(strnbins, "%d", nbins);
    sprintf(strkflux, "%d", kflux);

    // printf("nbins =%s \n",strnbins);




    strcpy(path_data,"../data/nbins=");
    strcat(path_data,strkernel);
    strcat(path_data,"/nbins=");
    strcat(path_data,strnbins);
    strcat(path_data,"/");
    strcat(path_data,strscheme);
    strcat(path_data,"/kmax=");
    strcat(path_data,strkflux);
    strcat(path_data,"/");
    // printf("path_data=%s \n",path_data);
    char command[200];
    strcpy(command,"[ ! -f ");
    strcat(command,path_data);
    strcat(command," ] && mkdir -p ");
    strcat(command,path_data);

    // printf("command =%s \n",command);
    system(command);


}


//function printf for quad precision
// void printf_qp(__float128 var){
//     unsigned int prec = 20;
//     unsigned int width = 46;

//     char buf[128];

//     unsigned int n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", width, var);
//     if ((size_t) n < sizeof buf){
//         printf ("%s", buf);
//     }
// }

