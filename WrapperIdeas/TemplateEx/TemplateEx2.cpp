//code for bubble sort using templates

#include<iostream>

template <class T>
void bubbleSort(T a[],int n){
    for(int i=0;i<n-1;i++)
        for(int j=n-1;i<j;j--)
            if(a[j],a[j-1])
                std::swap(a[j],a[j-1]);
}

int main(){
    int a[5]={10,50,30,40,20};
    int n = sizeof(a)/sizeof(a[0]);

    bubbleSort<int>(a,5);

    std::cout << "Sorted array : " << std::endl;
    for(int i=0; i<n; i++)
        std::cout << a[i] << " " << std::endl;

    return 0;

}