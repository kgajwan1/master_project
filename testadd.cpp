#include <bits/stdc++.h>
#include <ctime> 
#include<chrono>
//sequential pass part done..
int main()
{
	std::default_random_engine rng;
	std::chrono::time_point<std::chrono::system_clock> start, end; 
	rng.seed(time(0));
	std::uniform_int_distribution<int> nd(0, 1000000);
    	unsigned long long int sum = 0;
	/*size_t sum2;
	std::cout <<sizeof(sum);
	std::cout <<sizeof(sum2);
	printf("\nSIZE_MAX = %zu\n", SIZE_MAX);*/
	start = std::chrono::system_clock::now(); 
    	for (int i = 0;i<50000000;i++)
    	{
        int y = nd(rng);
        sum+=y;
    	}
        end = std::chrono::system_clock::now(); 

        std::cout <<"sum: "<<sum<<'\n';	
    std::chrono::duration<double> elapsed_seconds = end - start; 
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
 
    std::cout << "parallel finished computation at " << std::ctime(&end_time) 
              << "elapsed time: " << elapsed_seconds.count() << "s\n"; 

}
