#include<thread>
#include <omp.h>
#include <bits/stdc++.h> 
#include <ctime> 
#include <atomic>

constexpr long long firBound=  25000000;
constexpr long long secBound=  50000000;
constexpr long long thiBound=  75000000;
constexpr long long fouBound= 100000000;

void sumUp(std::atomic<unsigned long long> &sum, const std::vector<int> &v, unsigned long long start, unsigned long long end){
        unsigned int long long tmpSum{};
	for (auto i= start; i < end; ++i){
	    tmpSum += v[i];
    }
	    sum.fetch_add(tmpSum,std::memory_order_relaxed);
}
int main()
{
	//scrap
	/*size_t sum2;
	std::cout <<sizeof(sum);
	std::cout <<sizeof(sum2);
	printf("\nSIZE_MAX = %zu\n", SIZE_MAX);*/


	std::chrono::time_point<std::chrono::system_clock> start, end,start_1,end_1; 
	std::default_random_engine rng;
	rng.seed(time(0));
	std::uniform_int_distribution<int> nd(0, 1000000);
	//unsigned long long sum = 0;
	unsigned long long sum_1 = 0;
	std::atomic<unsigned long long> sum(0);
	

        std::atomic<unsigned long long> sum2{}; 	
	omp_lock_t writelock;
	omp_init_lock (&writelock);
	std::vector <int> v;
	for (int i = 0;i<500000000;i++)
    		{
			v.push_back(nd(rng));
    		}
  
	start_1 = std::chrono::system_clock::now(); 

    	for (int i = 0;i<500000000;i++)
    	{
                sum_1+=v[i];
    	}
        end_1 = std::chrono::system_clock::now(); 

        std::cout <<"sum, sequential: "<<sum_1<<'\n';	
    std::chrono::duration<double> elapsed_seconds_1 = end_1 - start_1; 
 
    std::cout << "elapsed time: " << elapsed_seconds_1.count() << "s\n"; 
    float mops3 = 500000000/elapsed_seconds_1.count();
    std::cout <<mops3<<'\n'; 

//OpenMp
    start = std::chrono::system_clock::now(); 
#pragma omp parallel for
    	for (auto i = 0;i<500000000;i++)
    	{
		sum.fetch_add(v[i]);

    	}
        std::cout <<"sum: "<<sum<<'\n';

    end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end - start; 
    std::cout << "elapsed time for openmp parallelism: " << elapsed_seconds.count() << "s\n"; 
    float mops2 = 500000000/elapsed_seconds.count();
    std::cout <<mops2<<'\n'; 

//c++11
    auto start_2 = std::chrono::system_clock::now();
    std::thread t1(sumUp,std::ref(sum2),std::ref(v),0,125000000);
    std::thread t2(sumUp,std::ref(sum2),std::ref(v),125000000,250000000);
    std::thread t3(sumUp,std::ref(sum2),std::ref(v),250000000,375000000);
    std::thread t4(sumUp,std::ref(sum2),std::ref(v),375000000,500000000);   

    t1.join();
    t2.join();
    t3.join();
    t4.join();	
    std::chrono::duration<double> time_duration= std::chrono::system_clock::now() - start_2;
    std::cout << "Time for c++11 addition " << time_duration.count() << " seconds" << std::endl;
    std::cout << "Result: " << sum2 << std::endl;
    float mops = 500000000/time_duration.count();
    std::cout <<mops; 
    std::cout << std::endl;
}
/*
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <queue>

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <vector>

#include <thread>
#include <stdlib.h>
#include <mutex>
#include <iterator>
#include <memory>

#include <utility>
#include <iomanip>

using namespace std;
using namespace std :: chrono;

// uint64_t  credited to sir on his mail
// Global Variables sue me

uint64_t NUMBER_OF_QUERIES = 0;
uint64_t QUERY_DIMENSIONS= 0;
uint64_t K = 0;
unsigned int number_of_threads = 0;
unsigned int MAX_POSSIBLE_THREADS = 0;
vector<uint64_t> NUMBER_OF_QUERIES_per_thread;
vector<thread> threads_for_tree;
uint64_t TF_ID= 0;
uint64_t QF_ID= 0;
uint64_t TRAIN_POINTS = 0;
uint64_t TRAIN_DIMENSIONS= 0;

// NO VECTOR OF VECTORS, I SUCK AT THAT, MAKE POINTS NOT VECTORS  //SLOGAN 101
struct Point 
{	
		//struct packing
		vector<float> coordinates;
		~Point() {}
};

//QUEUE TO HELP WITH KNN
class PriorityQueue
{
	public:
		PriorityQueue(uint64_t sz) : fixed_size{sz} {}
		~PriorityQueue() {}
		
		//stanford notes referred for help in priority queue
		void insert_if_nearer(pair<Point*, float>);
		
		static bool compare_distance(pair<Point*, float> p1, pair<Point*, float> p2)
		{
			//figured out this is better than if else
			return p1.second < p2.second;
		}

		bool full_checker() 
		{ 
			//stop when you have found enough neighbors
			return neighbours.size() < K; 
		}
		//sudo apt get neighbours
		vector<pair<Point*, float>> get_neighbours() 
		{ 
			return neighbours; 
		}

		uint64_t size() 
		{ 
			return neighbours.size(); 
		}

		float get_maximum() 
		{
			return neighbours[neighbours.size()-1].second; 
		}

		//no modifying these values
	private:
		uint64_t fixed_size;
		vector<pair<Point*, float>> neighbours;
		
};

void PriorityQueue::insert_if_nearer(pair<Point*, float> temp)
{
	// If Queue full, check to see if new point is smaller than biggest entry
	if (this->neighbours.size() == this->fixed_size)
	{
		if (temp.second < neighbours[fixed_size-1].second)
		{
			neighbours[fixed_size-1] = temp;
			sort(this->neighbours.begin(), this->neighbours.end(), this->compare_distance);
		}
	}

	else
	{
		this->neighbours.push_back(temp);
		sort(this->neighbours.begin(), this->neighbours.end(), this->compare_distance);
	}
}

// k-dimension tree, cmu notes
//stanford notes too
class Node {
    //make it accessible
    public:
		
        Node(Point* p) : point{p}, left_child{nullptr}, right_child{nullptr} {}
		~Node() {}
		vector<float> get_coordinates() 
		{ 	
			return point->coordinates; 
		}
		float get_coordinates(uint64_t i) 
		{
			return point->coordinates[i]; 
		}
		Node* get_left() 
		{ 
			return left_child; 
		}
		Node* get_right() 
		{ 
			return right_child; 
		}
		void insert_left(Node* child) 
		{ 
			this->left_child = child; 
		}
		void insert_right(Node* child) 
		{ 
			this->right_child = child;
		}
		void query(Point *q, Node *n, int dimension, PriorityQueue* cur_neighbours);

	//aint nobody gonna access that 
	private:
        	Point* point;
			Node *left_child;
			Node *right_child;
};

// Implement class to allow passing dimension for comparison

class compare
{
	public:
		compare(int d) : dimension(d) {}
		bool operator()(Point* p1, Point* p2) 
		{
			return p1->coordinates[dimension] < p2->coordinates[dimension];
    	}
	private:
    	int dimension;
};

// Some useful tools

//start with reading training file
shared_ptr<vector<Point*>> read_training_file(char* filename)
{
	static char buffer[8] = {}; // bufferer for reading data from file
	ifstream training_file (filename, ios::in|ios::binary); // open in binary mode

	// Invalid file handling
	if(!training_file)
	{
		cout << "No file named " << filename << endl;
		exit(1);
	}

	cout << "_______TRAINING_FILE RESULTS______" << endl;
	// Read in header data from file, 8*8 == 64
	training_file.read(buffer, 8);
	string filetype(buffer);
	
	training_file.read(buffer, 8);
	string fid(buffer);
	TF_ID= *((uint64_t*) buffer);
	cout << "TRAINING_FILE_ID: " << TF_ID<< endl;

	training_file.read(buffer, 8);
	TRAIN_POINTS = *((uint64_t*) buffer);
	cout << "TRAIN_POINTS: " << TRAIN_POINTS << endl;

	training_file.read(buffer, 8);
	TRAIN_DIMENSIONS= *((uint64_t*) buffer);
	cout << "TRAIN_DIMENSIONS: " << TRAIN_DIMENSIONS<< endl;

	// Create vector of size # training points
	shared_ptr<vector<Point*>> data (new vector<Point*>());

	// For every point, get dimension, then add it to vector. struct packing
	for(uint64_t i = 0; i < TRAIN_POINTS; i++)
	{
		Point* temp = new Point();
		for(uint64_t j = 0; j < TRAIN_DIMENSIONS; j++)
		{
			// 8 * 4 == 32
			training_file.read(buffer, 4);
			temp->coordinates.push_back(*((float*) buffer));
		}
		data->push_back(temp);
	}
	training_file.close();
	cout << "Number of training points in vector: " << data->size() << endl;
	assert(data->size() == TRAIN_POINTS);
	return data;
}


//then you read query
vector<Point*> read_query_file(char* filename)
{
	static char buffer[8] = {}; // bufferer for reading data from file
	ifstream file (filename, ios::in|ios::binary); // open in binary mode

	// Invalid file handling
	if(!file)
	{
		cout << "No file named " << filename << endl;
		exit(1);
	}

	cout << "\n_____QUERY_FILE RESULTS____" << endl;
	// Read in header data from file
	file.read(buffer, 8);
	string filetype(buffer);
	cout << filetype << endl;

	file.read(buffer, 8);
	QF_ID= *((uint64_t*) buffer);
	cout << "QUERY FILE ID: " << QF_ID<< endl;

	file.read(buffer, 8);
	NUMBER_OF_QUERIES = *((uint64_t*) buffer);
	cout << "NUMBER_OF_QUERIES: " << NUMBER_OF_QUERIES << endl;

	file.read(buffer, 8);
	QUERY_DIMENSIONS= *((uint64_t*) buffer);
	cout << "QUERY_DIMENSIONS: " << QUERY_DIMENSIONS<< endl;
	assert(QUERY_DIMENSIONS== TRAIN_DIMENSIONS);

	file.read(buffer, 8);
	K = *((uint64_t*) buffer);
	cout << "K: " << K << endl;
	assert(K > 0);

	// Create vector of size # training points
	vector<Point*> data;

	// For each training point, get each dimension, then add it to vector
	for(uint64_t i = 0; i < NUMBER_OF_QUERIES; i++)
	{
		Point* temp = new Point();
		for(uint64_t j = 0; j < QUERY_DIMENSIONS; ++j)
		{
			// Read each 32bit float coordinate dimension
			file.read(buffer, 4);
			temp->coordinates.push_back(*((float*) buffer));
		}
		data.push_back(temp);
	}
	file.close();

	cout << "Number of query points in vector: " << data.size() << endl;
	assert(data.size() == NUMBER_OF_QUERIES);
	return data;
}

//write output
void write_output_file(const char* filename, vector<vector<pair<Point*, float>>>* nearest_neighbours_vector)
{
	ofstream output_file(filename, ios::binary); // open in binary mode
	// file handler if file is invalid
	if(!output_file)
	{
		cout << "Not gonna write there... " << filename << endl;
		exit(1);
	}

	char filetype[8] = "RESULT";
	ifstream random_file ("/dev/urandom", ios::in|ios::binary); // open in binary mode
	static char buffer[8];
	random_file.read(buffer, 8);
	uint64_t result_file_ID = *((uint64_t*) buffer);

	output_file.write(reinterpret_cast<const char *> (&TF_ID), sizeof(TF_ID));
	output_file.write(reinterpret_cast<const char *> (&QF_ID), sizeof(QF_ID));

	output_file.write(reinterpret_cast<const char *> (&filetype), sizeof(filetype));
	output_file.write(reinterpret_cast<const char *> (&result_file_ID), sizeof(result_file_ID));
	output_file.write(reinterpret_cast<const char *> (&NUMBER_OF_QUERIES), sizeof(NUMBER_OF_QUERIES));

	output_file.write(reinterpret_cast<const char *> (&TRAIN_DIMENSIONS), sizeof(TRAIN_DIMENSIONS));
	output_file.write(reinterpret_cast<const char *> (&K), sizeof(K));

	for (uint64_t q = 0; q < NUMBER_OF_QUERIES; q++)
	{
		for(uint64_t i = 0; i < K; ++i)
		{
			for(uint64_t j = 0; j < QUERY_DIMENSIONS; ++j)
			{
				output_file.write(reinterpret_cast<const char *> (&((*nearest_neighbours_vector)[q][i].first->coordinates[j])), sizeof(float));
			}
		}
	}
	output_file.close();
}

// Returns index of median along given dimension
int find_median(shared_ptr<vector<Point*>> subarray, int dimension)
{
	if (subarray->size() > 1)
	{
		// Sort points in array by the values of the current dimension
		sort(subarray->begin(), subarray->end(), compare(dimension));
	}
	if (subarray->size() % 2 == 0)
	{
		return (subarray->size() / 2) - 1;
	}
	else
	{
		return (subarray->size() - 1) / 2;
	}
}

void build_left_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_right_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_left_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);
void build_right_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension);


void build_left_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}

	int median_index = find_median(data, dimension);
	assert(median_index >= 0);

	Point* median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as left child to parent
	Node* split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMENSIONS);
	build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);

}

void build_right_non_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);
	
	median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	// Add median point as right child to parent
	split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMENSIONS);
	build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);
}

void build_left_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	mutex mut_x;

	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);

	median_point = (*data)[median_index];
	assert(median_point != nullptr);

	// Add median point as left child to parent
	split = new Node(median_point);
	parent->insert_left(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin()+median_index+1, data->end()));

	// Lock mutex 

	unique_lock<mutex> lck(mut_x);
	if (number_of_threads < MAX_POSSIBLE_THREADS)
	{
		threads_for_tree.push_back(thread(build_left_threaded, split, left, (dimension+1) % TRAIN_DIMENSIONS));
		number_of_threads++;
		lck.unlock();
		build_right_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);
	}
	else
	{
		lck.unlock();
		build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMENSIONS);
		build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);
	}
}

void build_right_threaded(Node* parent, shared_ptr<vector<Point*>> data, int dimension)
{
	if (data == nullptr || data->size() == 0)
	{
		return;
	}
	mutex mut_x;

	int median_index;
	Point* median_point;
	Node* split;

	median_index = find_median(data, dimension);
	assert(median_index >= 0);

	median_point = (*data)[median_index];
	assert(median_point != nullptr);
	
	split = new Node(median_point);
	parent->insert_right(split);

	shared_ptr<vector<Point*>> left (new vector<Point*>(data->begin(), data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(data->begin() + median_index+1, data->end()));

	// Lock mutex to prevent access to critical region 
	unique_lock<mutex> lck(mut_x);
	if (number_of_threads < MAX_POSSIBLE_THREADS)
	{
		threads_for_tree.push_back(thread(build_left_threaded, split, left, (dimension+1) % TRAIN_DIMENSIONS));
		++number_of_threads;
		lck.unlock();
		build_right_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);
	}
	
	else
	{
		lck.unlock();
		build_left_non_threaded(split, left, (dimension+1) % TRAIN_DIMENSIONS);
		build_right_non_threaded(split, right, (dimension+1) % TRAIN_DIMENSIONS);
	}
}

//  delete all nodes
void delete_kd_tree(Node* current)
{
	if (current != nullptr)
	{
		delete_kd_tree(current->get_left());
		delete_kd_tree(current->get_right());
		delete current;
	}
}

// Traverse tree to print all nodes
void print_kd_tree(Node* current, string dir, int level)
{
	cout << dir << level << ": ";
	for(uint64_t j = 0; j < TRAIN_DIMENSIONS; ++j)
	{
		 cout << current->get_coordinates(j) << "  ";
	}
	cout << endl;

	if (current->get_left() != nullptr)
	{
		print_kd_tree(current->get_left(), "LEFT", level+1);
	}
	if (current->get_right() != nullptr)
	{
		print_kd_tree(current->get_right(), "RIGHT", level+1);
	}
}

float distance(Point* p1, Point* p2)
{
	float sum = 0.0;
	float difference = 0.0;
	for (uint64_t i = 0; i < QUERY_DIMENSIONS; ++i)
	{
		difference = p1->coordinates[i] - p2->coordinates[i];
		sum += difference*difference;
	}
	return sqrt(sum);
}

void Node::query(Point *q, Node *n, int dimension, PriorityQueue* cur_neighbours) 
{
	if (n == nullptr) 
	{
		return;
	}

	// Index of next dimension to consider (one level down).
	int next_dim = (dimension+ 1) % TRAIN_DIMENSIONS;

	float d = distance(q, n->point);
	cur_neighbours->insert_if_nearer(make_pair(n->point, d));

	if (q->coordinates[dimension] <= n->point->coordinates[dimension]) 
	{
		query(q, n->left_child, next_dim, cur_neighbours);
		
		
		if (n->point->coordinates[dimension] - q->coordinates[dimension] < cur_neighbours->get_maximum() ||
			cur_neighbours->full_checker()) 
		{
			query(q, n->right_child, next_dim, cur_neighbours);
		}
	} 

	else 

	{
		query(q, n->right_child, next_dim, cur_neighbours);
		if (q->coordinates[dimension] - n->point->coordinates[dimension] < cur_neighbours->get_maximum() ||
			cur_neighbours->full_checker()) 
		{
			query(q, n->left_child, next_dim, cur_neighbours);
		}
	}
}

void dispatch_query_threads(int thread_num, vector<Point*>* query_data, Node* root, vector<PriorityQueue*>* pq_vector, 
							vector<vector<pair<Point*, float>>>* nearest_neighbours_vector)
{
	uint64_t queries_to_do = NUMBER_OF_QUERIES_per_thread[thread_num];
	uint64_t base_value = 0;
	for (int i = 0; i < thread_num; ++i)
	{
		base_value += NUMBER_OF_QUERIES_per_thread[i];
	}
	for (uint64_t q = 0; q < queries_to_do; ++q)
	{
		uint64_t index = base_value + q;
		(*pq_vector)[index] = new PriorityQueue(K);
		root->query((*query_data)[index], root, 0, (*pq_vector)[index]);
	 	(*nearest_neighbours_vector)[index] = (*pq_vector)[index]->get_neighbours();
	}
}

int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << "Error: Incorrect execution "<< endl; 
		cout << "Execution format: ./a.out <num_cores> <training_file> <query_file> <results_file>" << endl;
		exit(1);
	}
	// Assign command-line args
	int num_cores = atoi(argv[1]); 
	assert(num_cores > 0);
	MAX_POSSIBLE_THREADS = num_cores;
	char* training_file = argv[2];
	char* query_file = argv[3];
	char* results_file = argv[4];

	// READ TRAINING FILE
	auto start = chrono::high_resolution_clock::now();

	shared_ptr<vector<Point*>> training_data = read_training_file(training_file);	

	auto stop = chrono::high_resolution_clock::now();
	chrono::duration<double> dt = stop - start;
	cout << "Time for reading training file: " << dt.count() << endl;
	
	// READ QUERY FILE
	start = chrono::high_resolution_clock::now();
	
	vector<Point*> query_data = read_query_file(query_file);
	
	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time for reading query file: " << dt.count() << endl;
	
	// CHECKING TRAINING FILE RESULTS
	assert(training_data->size() == TRAIN_POINTS);

	// Build K-D Tree
	start = chrono::high_resolution_clock::now();
	
	int median_index = find_median(training_data, 0);
	Point* median_point = (*training_data)[median_index];
	Node* root = new Node(median_point);
	
	shared_ptr<vector<Point*>> left (new vector<Point*>(training_data->begin(), training_data->begin()+median_index));
	shared_ptr<vector<Point*>> right (new vector<Point*>(training_data->begin()+median_index+1, training_data->end()));

	// Call subfunctions to split each half of the training_data
	number_of_threads = 1;
	threads_for_tree.push_back(thread(build_left_threaded, root, left, 1 % TRAIN_DIMENSIONS));
	build_right_threaded(root, right, 1 % TRAIN_DIMENSIONS);
	
	// Handle cleaning up the threads
	cout << "\n____BUILDING K-D TREE____" << endl;

	while (number_of_threads != MAX_POSSIBLE_THREADS && number_of_threads != threads_for_tree.size())
	{
		continue;
	}
	
	cout << "All threads created!" << endl;
	cout << "Joining Threads now..." << endl;

	for (unsigned int i = 0; i < MAX_POSSIBLE_THREADS; ++i)
	{
		//cout << "Joining thread " << i+1 << "/" << MAX_POSSIBLE_THREADS << endl;
		if (threads_for_tree[i].joinable())
		{
			threads_for_tree[i].join();
		}
	}

	//jedi, I am
	cout << "All joined, threads are" << endl;

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to build Tree: " << dt.count() << endl;

	cout << "\n____QUERYING TREE____" << endl;
	// CHECKING FILE DATA
	cout << "Number of query points in vector: " << query_data.size() << endl;
	assert(query_data.size() == NUMBER_OF_QUERIES);

	// vec of vectors. Just hate doing this, but No dynamic arrays now
	vector<vector<pair<Point*, float>>> nearest_neighbours_vector(NUMBER_OF_QUERIES);	

	// Priority Queue of pairs base_valued on comparator for distances in pair
	vector<PriorityQueue*> pq_vector(NUMBER_OF_QUERIES);
	
	// Get start time for all queries
	start = chrono::high_resolution_clock::now();

	// IF MORE THREADS THAN QUERIES, ONLY MAKE 1 THREAD FOR EACH
	if (MAX_POSSIBLE_THREADS > NUMBER_OF_QUERIES)
	{
		MAX_POSSIBLE_THREADS = NUMBER_OF_QUERIES;
	}

	// Initialize to most even distribution of queries
	NUMBER_OF_QUERIES_per_thread = vector<uint64_t>(MAX_POSSIBLE_THREADS, (int) NUMBER_OF_QUERIES/MAX_POSSIBLE_THREADS);
	// Get remainder of the division
	uint64_t temp = NUMBER_OF_QUERIES % MAX_POSSIBLE_THREADS;
	unsigned int i = 0;
	// Add 1 to each thread's workload until remainder is gone
	while (temp != 0)
	{
		NUMBER_OF_QUERIES_per_thread[i]++;
		temp--;
		i++;
	}

	start = chrono::high_resolution_clock::now();
 
	vector<thread> query_threads;
	for (i = 0; i < MAX_POSSIBLE_THREADS; i++)
	{
		query_threads.push_back(thread(dispatch_query_threads, i, &query_data, root, &pq_vector, &nearest_neighbours_vector));
		this_thread::sleep_for(chrono::milliseconds(1));
	}

	for (i = 0; i < MAX_POSSIBLE_THREADS; i++)
	{
		query_threads[i].join(); // Wait until thread is done
	}
	
	stop = chrono::high_resolution_clock::now();
	dt = stop - start;

	cout << "Time to query KD-Tree for all queries: " << dt.count() << endl;

	cout << "\n____RESULTS FILE____" << endl;
	cout << "Writing results file..." << endl;

	start = chrono::high_resolution_clock::now();

	write_output_file(results_file, &nearest_neighbours_vector);

	stop = chrono::high_resolution_clock::now();
	dt = stop - start;
	cout << "Time to write results file: " << dt.count() << endl;


	// CLEAN UP REMAINING POINTERS
	cout << "\n____CLEANING UP____" << endl;
	cout << "Deleting training data..." << endl;
	for (uint64_t i = 0; i < TRAIN_POINTS; ++i)
	{
		delete (*training_data)[i];
	}

	cout << "Deleting query and nearest neighbor data..." << endl;
	for (uint64_t i = 0; i < NUMBER_OF_QUERIES; ++i)
	{
		delete query_data[i];
		delete pq_vector[i];
		
	}

	cout << "Deleting tree data...\n" << endl;
	delete_kd_tree(root);
	return 0;

}*/