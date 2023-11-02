#include<stdio.h>
#include<math.h>
#include<conio.h>

#define entrada 3
#define saida 2
#define in 4

int main(int argc, char const *argv[])
{
    float w[entrada][saida], err ,erro[saida], ni[saida], errom, bias, eta ,entradas[in][saida], saidas[in][saida], phi[saida];

    int x, cont, contt, contin=0, epocas, testeerro=0, funcao;
    char continua = 's';
    for (x = 0; x< entrada; x++) 
        for (cont = 0; cont<saida; cont++) 
            w[x][cont] = 0;
    
    // testar no linux
    // clrscr(); 


    printf ("Entre com o valor do bias: ");
    scanf ("Sf", &bias);
    printf ("Entre com o valor da taxa de aprendizagem: ");
    scanf ("&f", &eta);
    printf ("Entre com o número de iterações: ");
    scanf ("°d", &epocas);
    printf ("Entre com o valor do erro esperado: ");
    scanf ("&f", &err);
    printf ("Entre com a função desejada [ (1) degrau, (2) sigmoide]: ");
    scanf ("8d", &funcao);
    printf ("Entre com os dados de entrada e de saída para o treinamento: \n");

    for (x = 0; x < in; x++)
        for (cont = 0; cont < saida; cont++) {
            printf ("Entrada %d, Neurônio %d:", x + 1, cont + 1);
            scanf ("%f", &entradas [x] [cont]);
        }
    return 0;
}
