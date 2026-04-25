module test_mem #(
    parameter SIZE = 1024
);
    // Case 1: Parameterized Range with 0:
    reg [7:0] mem1 [0:SIZE-1];
    
    // Case 2: Parameterized Size only
    // reg [7:0] mem2 [SIZE-1:0]; 
    
    // Case 3: Fixed Size
    // reg [7:0] mem3 [0:1023];
endmodule
