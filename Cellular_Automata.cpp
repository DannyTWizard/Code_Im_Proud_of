#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <bitset>
#include <random>





 void error_callback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;

}




// Draw a black/white strip starting 'topOffsetPx' pixels below the top edge.
void drawBWStrip(GLFWwindow* window,
                 const std::vector<int>& bits,
                 int bandHeightPx,
                 int topOffsetPx) {
    if (bits.empty() || bandHeightPx <= 0) return ;

    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    if (fbW <= 0 || fbH <= 0) return ;

    const int n = (int)bits.size();
    const int wpx = fbW / n;

    // y origin is bottom-left; top of strip:
    int y = fbH - bandHeightPx - topOffsetPx;
    if (y < 0) return ; // fully off-screen

    glEnable(GL_SCISSOR_TEST);

    int x = 0;
    for (int i = 0; i < n; ++i) {
        int thisW = (i == n - 1) ? (fbW - x) : wpx;

        if (bits[i]) glClearColor(1.f, 1.f, 1.f, 1.f); // white
        else         glClearColor(0.f, 0.f, 0.f, 1.f); // black

        glScissor(x, y, thisW, bandHeightPx);
        glClear(GL_COLOR_BUFFER_BIT);

        x += thisW;
    }

    glDisable(GL_SCISSOR_TEST);
    
}



std::bitset<8> decimalToBinary(int n) {
    return std::bitset<8>(n);
}

//now we want to create a function that will convert fill out the if statements for the 8 possible combinations of 3 bits

std::vector<int> applyRule(std::vector<int>& cells, int ruleNumber){
    std::bitset<8> rule = decimalToBinary(ruleNumber);
    std::vector<int> newCells(cells.size(),0);
    int size = cells.size();
    for (int i = 0; i < size; ++i) {
        int left = cells[(i - 1 + size) % size]; // wrap around
        int center = cells[i];
        int right = cells[(i + 1) % size]; // wrap around

        if (left == 1 && center == 1 && right == 1) {
            newCells[i] = rule[7];
            //std::cout << "111 -> " << rule[0] << std::endl;
        } else if (left == 1 && center == 1 && right == 0) {
            newCells[i] = rule[6];
            //std::cout << "110 -> " << rule[1] << std::endl;
        } else if (left == 1 && center == 0 && right == 1) {
            newCells[i] = rule[5];
            //std::cout << "101 -> " << rule[2] << std::endl;
        } else if (left == 1 && center == 0 && right == 0) {
            newCells[i] = rule[4];
            //std::cout << "100 -> " << rule[3] << std::endl;
        } else if (left == 0 && center == 1 && right == 1) {
            newCells[i] = rule[3];
            //std::cout << "011 -> " << rule[4] << std::endl; 
        } else if (left == 0 && center == 1 && right == 0) {
            newCells[i] = rule[2];
            //std::cout << "010 -> " << rule[5] << std::endl;
        } else if (left == 0 && center == 0 && right == 1) {
            newCells[i] = rule[1];
            //std::cout << "001 -> " << rule[6] << std::endl;
        } else if (left == 0 && center == 0 && right == 0) {
            newCells[i] = rule[0];
            //std::cout << "000 -> " << rule[7] << std::endl;

        }


    }

    return newCells;


} 


//We need to specify the array length
//the rule number
//the bar size

// Initialize GLFW and create a window
int main() {
    /// This iniialises the glfw window that you are going to be editing
    /// I assume this also cha

    int RULE;

    std::cout<<"Hello today we are going to animate some cellular automata"<<std::endl;
    std::cout<<"what rule number to you want to use "<<std::endl;
    std::cin>>RULE;
    std::cout<<"you have chosen rule number: "<<RULE<<"excellent choice!"<<std::endl;
    std::bitset<8> rulebit = decimalToBinary(RULE);
    std::cout<<"end of rule"<<rulebit[0]<<std::endl;
    std::cout<<"which is binary: "<<rulebit<<std::endl;

    if (RULE<0 || RULE>255){
        std::cout<<"that is not a valid rule number, please choose a number between 0 and 255"<<std::endl;
        return -1;
    }

    std::cout<<"how many cells do you want in your array? (recommend 10-200)"<<std::endl;
    int array_length=0;
    std::cin>>array_length; 

    std::cout<<"you have chosen an array length of: "<<array_length<<std::endl;


    if (!glfwInit()) {
        return -1;
    }

    int framebufferWidth {800};
    int framebufferHeight {600};

    ///This creates the window (maybe as a pointer, presumably the function create window outputs an address)

    ///If the window creation fails we terminate the glfw session
    GLFWwindow* window = glfwCreateWindow(framebufferWidth, framebufferHeight, "OpenGL Example", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    /// This makes the window you have created the current context
    glfwMakeContextCurrent(window);


    /// Initialize GLEW
    /// This loads the pointers to all the open gl functions that live on the driver
    /// @brief Initialize GLEW
    if (glewInit() != GLEW_OK) {
        return -1;
    }

    

    //std::vector<int> bitsrow1 = {1,0,1,1,0,0,1,0,1,0}; // example pattern
    //std::vector<int> bitsrow2 = {0,1,0,0,1,1,0,1,0,1}; // example pattern

    //const int n = (int)bitsrow1.size();
    //rows.push_back(bitsrow1);
    //rows.push_back(bitsrow2);
    //rows.push_back(bitsrow1);


    int bandHeightPx = framebufferWidth/array_length;
    /// Main loop
    /// The main rendering loop runs until you close the window
    ///Inside this loop is where all the draw commands will be issued

    double lastStepTime = glfwGetTime();
    
    const double stepInterval = 0.5; // seconds between revealing rows

    double nextStepTime = glfwGetTime() + stepInterval;

    size_t rowcounter = 0;

    const int row_nums= std::floor(framebufferHeight/bandHeightPx);

    std::cout<<"row nums: "<<row_nums<<std::endl;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::bernoulli_distribution dist(0.5);  
        
    // std::vector<int> cells(array_length,0);
    // for(int i=0;i<array_length;i++){
    //     cells[i]=dist(gen);
    // }    

    while (!glfwWindowShouldClose(window)) {
    //while(true){

        ///A buffer is the current image open gl is drawing to
        /// this wipes the current buffer and resets it to the last set clear color
        glDisable(GL_SCISSOR_TEST);                 // make sure we’re clearing everything
        glClearColor(0.15f, 0.15f, 0.18f, 1.f);     // dark gray background
        glClear(GL_COLOR_BUFFER_BIT);
        // Draw the black/white squares band at the top

        //drawBWStrip(window, bitsrow1, bandHeightPx, 0);
        //rowcounter = 1;

        





        double now = glfwGetTime();
        int ticks = (int)std::floor((now - lastStepTime) / stepInterval);

        //std::vector<std::vector<int>> rows;

        std::deque<std::vector<int>> rows;

        std::vector<int> cells(array_length,0);
        cells[array_length/2]=1; //set the middle cell to 1

        // cells[0]=1; //set the first cell to 1
        // cells[30]=1; //set the 30th cell to 1
        // cells[60]=1; //set the 60th cell to 1
        // cells[90]=1; //set the 90th cell to 1
        // cells[137]=1;
        // cells[40]=1;
        // cells[50]=1;
        // cells[51]=1;
        // cells[52]=1;

        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::bernoulli_distribution dist(0.5);  
        
        // std::vector<int> cells(array_length,0);
        // for(int i=0;i<array_length;i++){
        //     cells[i]=dist(gen);
        // }

        for(size_t i{0}; i< ticks; ++i){

            std::vector<int> r=applyRule(cells,RULE);
            rows.push_back(r);
            cells=r;
        }

        if (ticks > row_nums) {

            int excess = ticks - row_nums;

            for(size_t i {0};i<excess;i++){
                rows.pop_front();
            }

        }

        std::cout << "deque size: " << rows.size() << std::endl;

        size_t rowcounter=0;
        for(auto row :rows){
            drawBWStrip(window, row, bandHeightPx, (rowcounter * bandHeightPx));
            rowcounter++;
            //std::cout << "W: " << W[0] << std::endl;
            //std::cout << "wpx: " << W[1] << std::endl;
        }
        

        // if (ticks % 2 == 0){
        //     for(size_t i{0}; i< ticks; ++i){
        //         drawBWStrip(window, bitsrow1, bandHeightPx, (i * bandHeightPx));

        //     }
        // }else{
        //     for(size_t i{0}; i< ticks; ++i){
        //         drawBWStrip(window, bitsrow2, bandHeightPx, (i * bandHeightPx));
        //     }
        // }

        // if (now >= nextStepTime) {
        //     drawBWStrip(window, bitsrow1, bandHeightPx, (rowcounter * bandHeightPx));
        //     nextStepTime += stepInterval;
        //     rowcounter++;

        // }

        // Draw the black/white squares band at the bottom
        //drawBWStrip(window, bitsrow2, bandHeightPx, bandHeightPx);

        /// Swap the buffers
        ///Modern graphics hold two buffers, the shown one and the next drawn one
        /// This swaps them to avoid flickering or tearing which would happen if you drew then loaded
        glfwSwapBuffers(window);

        // Poll for events
        ///This stops to take stock of any user inputs. Once registered the window can respond according to how you instruct it
        glfwPollEvents();
    }

    // Terminate GLFW
    glfwTerminate();
    return 0;


}