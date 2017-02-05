// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2014/2015, March 2 - April 3
// ###
// ### Project Name: Poisson Image Editing
// ### Group members: Gaurav Krishna Joshi, Saion Chatterjee
// ### 
// ### Supervisor: Thomas Moellenhoff
// ###
// ###
// ### Note for user: Please use same image format and same dimension for all the 3 input images- source, mask and target
// ###


#include "aux.h"
#include <iostream>
#include "math.h"
using namespace std;

//  uncomment to use the camera
//#define CAMERA

//  uncomment any one of the following to run the corresponding version- CPU, GAUSS, SOR or SHARED
//  WARNING: Do not uncomment multiple versions below. Only one of them must be uncommented.
//#define CPU
//#define GAUSS
#define SOR
//#define SHARED

//  uncomment any one of the following to run with corresponding guiding gradient- source or mixed or no guiding gradient
//  WARNING: Do not uncomment multiple versions below. Only one of them must be uncommented.
//#define SOURCE_GRADIENT
#define MIXED_GRADIENT
//#define NO_GRADIENT

//  Iteration value- can also be passed from the command line
#define ITERATIONS 7000

//  please do not comment the below parameters
#define THETA 0.9

#define INSIDE_MASK           0
#define BOUNDRY               1
#define OUTSIDE               2

#define CORNER_PIXEL_0_0      3
#define CORNER_PIXEL_0_H      4 
#define CORNER_PIXEL_W_0      5
#define CORNER_PIXEL_W_H      6 

#define EDGE_PIXEL_RIGHT      7 
#define EDGE_PIXEL_LEFT       8
#define EDGE_PIXEL_UP         9
#define EDGE_PIXEL_DOWN       10 


// Source Image Masking Kernel. 
__global__ void SourceImageMasking(float *Imgsrc, float *Mask, float *ImgOut_Srcmask, int nc, int w ,int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  for (int channel=0;channel < nc ;channel++)
	{
		int id=x + w*y + w*h*channel;

		//Normalizing mask id for any coloured pixels if left out during manual masking
		if(Mask[id]<0.5)
			Mask[id]=0;
		else
			Mask[id]=1;

        //Creating the image with only the masked portion
		ImgOut_Srcmask[id]=Imgsrc[id]*Mask[id];
	}   
}



// Extracting Boundry Pixel using the mask. 
__global__ void ExtractingBoundryPixels(float *Mask, int *BoundryPixelsArray,int nc, int w ,int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  for (int channel=0;channel < nc ;channel++)
	{
       if(x<w && y<h)
            {
			int id=x + w*y +w*h*channel;

			if(x==0 && y==0 && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = CORNER_PIXEL_0_0;  
			  }
			else if(x==0 && y==(h-1) && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = CORNER_PIXEL_0_H;  
			  }
			else if(x==(w-1) && y==0 && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = CORNER_PIXEL_W_0;  
			  }
			else if(x==(w-1) && y==(h-1) && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = CORNER_PIXEL_W_H;  
			  }
			else if(x==0 && y<(h-1) && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = EDGE_PIXEL_LEFT;  
			  }
			else if(x==(w-1) && y<(h-1) && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = EDGE_PIXEL_RIGHT;  
			  }
			else if(x<(w-1) && y==0 && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = EDGE_PIXEL_DOWN;  
			  }
			else if(x<(w-1) && y==(h-1) && Mask[id]==1)
			  {
			      BoundryPixelsArray[id] = EDGE_PIXEL_UP;  
			  }
                else
                  {  
			  int id_r=x+1 + w*y +w*h*channel;
			  int id_l=x-1 + w*y +w*h*channel;
			  int id_u=x + w*(y+1) +w*h*channel;
			  int id_d=x + w*(y-1) +w*h*channel;

			  if(Mask[id]==1 && Mask[id_r]==1 && Mask[id_l]==1 && Mask[id_u]==1 && Mask[id_d]==1)
				 {
				    BoundryPixelsArray[id]=INSIDE_MASK;   // Totally Inside 
				 }
			  else if((Mask[id]==1) && (Mask[id_r]==0 || Mask[id_l]==0 || Mask[id_u]==0 || Mask[id_d]==0))
				 {
				    BoundryPixelsArray[id]=BOUNDRY;  //Boundry
				 }
   	                  else
	                         {
	                            BoundryPixelsArray[id]=OUTSIDE;  //Totally outside
	                         }
                  }
            }
      }
}



//Calculate boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY the bound box variables for the selected region.
void calculate_boundBoxMinMax(int w, int h, int nc, int *BoundaryPixelsArray, int *boundBoxMinX, int *boundBoxMinY, int *boundBoxMaxX, int *boundBoxMaxY)
{
	//Initializing variables
	*boundBoxMinX=99999;
	*boundBoxMinY=99999;
	*boundBoxMaxX=0;
	*boundBoxMaxY=0;

	//Calculating boundBoxMinX and boundBoxMinY
	for(int c=0;c<nc;c++)
	{
		for(int x=0;x<=w-1;x++)
		{
			for(int y=0;y<=h-1;y++)
			{
				int idx = x + w*y +w*h*c;

				if(BoundaryPixelsArray[idx]==BOUNDRY)
				{
					  if(x < *boundBoxMinX)
					  {
					      *boundBoxMinX = x;
					  }
					  if(x > *boundBoxMaxX)
					  {
					       *boundBoxMaxX = x;
					  }
					  if(y < *boundBoxMinY)
					  {
					      *boundBoxMinY = y;
					  }
					  if(y > *boundBoxMaxY)
					  {
					       *boundBoxMaxY = y;
					  }          
		                }  
		        }
	       }
     }
}

//Test Kernel to verify image boundaries- This is only for testing purpose and so no need to call this.
__global__ void BoundryTest(float *Imgin, int *BoundryPixelsArray ,float *ImgOut, int nc, int w ,int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  for (int channel=0;channel < nc ;channel++)
    {
		int id=x + w*y +w*h*channel;

		if(BoundryPixelsArray[id]==BOUNDRY)
		{
			ImgOut[id]=1;
		}
		else
		{
			ImgOut[id]=Imgin[id];
		}
    }   
}

//Kernel to show NormalCloning. Simple copy paste of the selected region without any blending technique.
__global__ void SourceMaskImageMergeinTargetImage(float *Imgsrc, float *Imgtarget, float *ImgOut, int nc, int w ,int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  for (int channel=0;channel < nc ;channel++)
    {
		int id=x + w*y +w*h*channel;

        if(Imgsrc[id]==INSIDE_MASK)
           {
               ImgOut[id]=Imgtarget[id];
           }
        else
           {
               ImgOut[id]=Imgsrc[id];
           }
    }   
}

// Initializing the Output image.
__global__ void initialize(float *t, float *u, int *boundary, float *s, int w , int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    //Initialization for Dirichlet boundary conditions
   for(int c=0; c<nc; c++)
   {
		int idx = x + w*y +w*h*c;

		if(boundary[idx]==BOUNDRY || boundary[idx]==OUTSIDE) 
			u[idx]=t[idx];        // u is the final output image and t is the input target image
		else
			u[idx]=0;
	}
}

// Evaluating the Guiding Gradient of the output image's selected region
__global__ void evaluate_gradient(float *v_l, float *v_r, float *v_d, float *v_u, int *boundary, float *s, float *t, int target_nc, int w, int h)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;
    
   if(x<w && y<h)//Boundary of entire pixel grid
	   {
		  for(int c=0; c<target_nc; c++)
		  {
			  int idx = x + w*y +w*h*c;
			  int idx_nextX = x+1 + w*y +w*h*c;
			  int idx_prevX = x-1 + w*y + w*h*c;
			  int idx_nextY = x + w*(y+1) +w*h*c;
			  int idx_prevY = x + w*(y-1) +w*h*c; 

#ifdef NO_GRADIENT
			  //For No guiding gradient approach
			  v_r[idx] = 0;
			  v_l[idx] = 0;
			  v_u[idx] = 0;
			  v_d[idx] = 0;
#endif

#ifdef SOURCE_GRADIENT
	                 //For Source gradient approach

                         switch (boundary[idx]) //Required for handling for corner pixels of the image grid
		            {
					case INSIDE_MASK:  
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_u[idx] = s[idx]-s[idx_prevY];
						    v_d[idx] = s[idx]-s[idx_nextY];  
					break;
					case CORNER_PIXEL_0_0:
						    v_l[idx] = 0;
							v_u[idx] = 0;
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_d[idx] = s[idx]-s[idx_nextY];  

					break;      
					case CORNER_PIXEL_0_H:
							v_l[idx] = 0;
							v_d[idx] = 0;
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_u[idx] = s[idx]-s[idx_prevY];

					break;       
					case CORNER_PIXEL_W_0:
							v_r[idx] = 0;
							v_u[idx] = 0;
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_d[idx] = s[idx]-s[idx_nextY];  

					break;
					case CORNER_PIXEL_W_H: 
							v_r[idx] = 0;
							v_d[idx] = 0;
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_u[idx] = s[idx]-s[idx_prevY];

					break;

					case  EDGE_PIXEL_RIGHT:
							v_r[idx] = 0;
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_u[idx] = s[idx]-s[idx_prevY];
						    v_d[idx] = s[idx]-s[idx_nextY];  

					break;
					case  EDGE_PIXEL_LEFT:
							v_l[idx] = 0;
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_u[idx] = s[idx]-s[idx_prevY];
						    v_d[idx] = s[idx]-s[idx_nextY];  

					break;    
					case  EDGE_PIXEL_UP:
							v_d[idx] = 0;
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_u[idx] = s[idx]-s[idx_prevY];

					break;
					case  EDGE_PIXEL_DOWN:
							v_u[idx] = 0;
						    v_r[idx] = s[idx]-s[idx_nextX];
						    v_l[idx] = s[idx]-s[idx_prevX];
						    v_d[idx] = s[idx]-s[idx_nextY];  

				break; 
	    	   }
#endif  //Endif for Source gradient

#ifdef MIXED_GRADIENT
       //For Mixed gradient approach    
		
       //Declaring the source s and target t variables to store the relative gradient of the neighbouring pixels in 4-directions
		   float s_diff_r, t_diff_r;
		   float s_diff_l, t_diff_l;
		   float s_diff_d, t_diff_d;
		   float s_diff_u, t_diff_u;

       switch (boundary[idx]) //Required for handling for corner pixels of the image grid
	            {
					case INSIDE_MASK:  

							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = t[idx]-t[idx_nextY];  

					break;

					case CORNER_PIXEL_0_0:
							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = 0;
						    s_diff_u = 0;
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = 0;
						    t_diff_u = 0;
						    t_diff_d = t[idx]-t[idx_nextY];  

					break;      
					case CORNER_PIXEL_0_H:
							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = 0;
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = 0;

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = 0;
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = 0;

					break;       
					case CORNER_PIXEL_W_0:
							s_diff_r = 0;
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = 0;
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = 0;
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = 0;
						    t_diff_d = t[idx]-t[idx_nextY];  

					break;
					case CORNER_PIXEL_W_H: 
							s_diff_r = 0;
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = 0;

							t_diff_r = 0;
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = 0;

					break;

					case  EDGE_PIXEL_RIGHT:
						s_diff_r = 0;
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = 0;
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = t[idx]-t[idx_nextY];  

					break;
					case  EDGE_PIXEL_LEFT:
							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = 0;
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = 0;
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = t[idx]-t[idx_nextY];  

					break;    
					case  EDGE_PIXEL_UP:
							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = s[idx]-s[idx_prevY];
						    s_diff_d = 0;

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = t[idx]-t[idx_prevY];
						    t_diff_d = 0;

					break;
					case  EDGE_PIXEL_DOWN:
							s_diff_r = s[idx]-s[idx_nextX]; 
						    s_diff_l = s[idx]-s[idx_prevX];
						    s_diff_u = 0;
						    s_diff_d = s[idx]-s[idx_nextY];  

							t_diff_r = t[idx]-t[idx_nextX]; 
						    t_diff_l = t[idx]-t[idx_prevX];
						    t_diff_u = 0;
						    t_diff_d = t[idx]-t[idx_nextY];  

					break; 
	    	   }

//After appropriate evaluation of relative gradient of s and t, now comparing and assigning greater abs value for mixed gradient approach

						  if(abs(s_diff_r) > abs(t_diff_r))
						  	v_r[idx] = s_diff_r;
						  else
							v_r[idx] = t_diff_r;

						  if(abs(s_diff_d) > abs(t_diff_d))
							 v_d[idx] = s_diff_d;
						  else
							 v_d[idx] = t_diff_d;

						  if(abs(s_diff_l) > abs(t_diff_l))
							 v_l[idx] = s_diff_l;
						  else
							 v_l[idx] = t_diff_l;

						  if(abs(s_diff_u) > abs(t_diff_u))
							 v_u[idx] = s_diff_u;
						  else
							 v_u[idx] = t_diff_u;

#endif // endif for mixed gradient
		        
			 }
	   }

}


// Gauss Seidel implementation- GPU
__global__ void poisson_gauss_seidel(int *boundary, float *u, int w , int h,int target_nc,int boundBoxMinX, int boundBoxMinY, int boundBoxMaxX, int boundBoxMaxY, float *v_l, float *v_r, float *v_d, float *v_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x + boundBoxMinX;
    int y = threadIdx.y + blockDim.y * blockIdx.y + boundBoxMinY;
    
	if(x<=boundBoxMaxX && y<=boundBoxMaxY)//Boundary of selected bound box region
  	 {
  	for(int c=0; c<target_nc; c++)
		{
			int idx = x + w*y +w*h*c;
			int idx_nextX = x+1 + w*y +w*h*c;
			int idx_prevX = x-1 + w*y + w*h*c;
			int idx_nextY = x + w*(y+1) +w*h*c;
			int idx_prevY = x + w*(y-1) +w*h*c;  
          
        switch(boundary[idx]) 
           {
                case INSIDE_MASK  :
	   	      u[idx] = 0.25*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
        		break; 
		case CORNER_PIXEL_0_0:
			u[idx] = 0.5*(u[idx_nextX]+u[idx_nextY]+v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break;      
		case CORNER_PIXEL_0_H:
			u[idx] = 0.5*(u[idx_nextX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break;        
		case CORNER_PIXEL_W_0:
			u[idx] = 0.5*(u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
		case CORNER_PIXEL_W_H: 
			u[idx] = 0.5*(u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
		case  EDGE_PIXEL_RIGHT:
			u[idx] = 0.33*(u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
		case  EDGE_PIXEL_LEFT:
			u[idx] = 0.33*(u[idx_nextX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
		case  EDGE_PIXEL_UP:
			u[idx] = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
		case  EDGE_PIXEL_DOWN:
			u[idx] = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
			break; 
          }
	}
  }
}

//Copying Image to solution- needed for SOR implementation
__global__ void CopyImageoutinSolution(float *Imgout, float *solution, int nc, int w ,int h)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
  	for(int c=0; c<nc; c++)
		{
			int idx = x + w*y +w*h*c;
			solution[idx]=Imgout[idx];
       }
}

//SOR redblack implementation- GPU
__global__ void poisson_sor_redblack(int *boundary, float *u, float *solution, int w , int h, int target_nc, int boundBoxMinX, int boundBoxMinY, int boundBoxMaxX, int boundBoxMaxY, float *v_l, float *v_r, float *v_d, float *v_u, int param)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x + boundBoxMinX;
    int y = threadIdx.y + blockDim.y * blockIdx.y + boundBoxMinY;

    float result=0; //Using local variables for storing intermediate values to reduce array lookup for optimization purpose
    
    float theta = THETA;
    
if(x<=boundBoxMaxX && y<=boundBoxMaxY)//Boundary of selected bound box region
   {
  	for(int c=0; c<target_nc; c++)
	      {
		int idx = x + w*y +w*h*c;
		int idx_nextX = x+1 + w*y +w*h*c;
		int idx_prevX = x-1 + w*y + w*h*c;
		int idx_nextY = x + w*(y+1) +w*h*c;
		int idx_prevY = x + w*(y-1) +w*h*c; 

	    switch(boundary[idx])
	       {
                case INSIDE_MASK  :

		if((x+y)%2 == 0 && param == 0 /*Red step*/)
			{ 
				result = 0.25*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
				solution[idx] =  result + theta*(result - solution[idx]) ;
			        // storing the result in the current pixel Optimization
			        u[idx]=result;
			}
		if((x+y)%2 == 1 && param == 1 /*Black step*/)
			{   
				result = 0.25*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
				solution[idx] =  result + theta*(result - solution[idx]) ;    
			        // storing the result in the current pixel Optimization
			        u[idx]=result;
			}

			break; 

		case CORNER_PIXEL_0_0:

			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.5*(u[idx_nextX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.5*(u[idx_nextX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}

   		        break;   

		case CORNER_PIXEL_0_H:

			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.5*(u[idx_nextX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.5*(u[idx_nextX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
		    break;      

		case CORNER_PIXEL_W_0:

			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.5*(u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.5*(u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
				       // storing the result in the current pixel Optimization
				       u[idx]=result;  
				}
		    break;

		case CORNER_PIXEL_W_H: 
			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.5*(u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				       // storing the result in the current pixel Optimization
				       u[idx]=result;  
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.5*(u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
				       // storing the result in the current pixel Optimization
				       u[idx]=result;  
				}
		    break;

		case  EDGE_PIXEL_RIGHT:
			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.33*(u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
		       		       // storing the result in the current pixel Optimization
		       		       u[idx]=result; 	
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.33*(u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
		       		       // storing the result in the current pixel Optimization
		       		       u[idx]=result; 							
				}

		    break;

		case  EDGE_PIXEL_LEFT:
			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.33*(u[idx_nextX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result= 0.33*(u[idx_nextX]+u[idx_nextY]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;    
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}
		    break;      

		case  EDGE_PIXEL_UP:
			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				        // storing the result in the current pixel Optimization
				        u[idx]=result; 
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_prevY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;   
				        // storing the result in the current pixel Optimization
				        u[idx]=result;  
				}

		    break;

		case  EDGE_PIXEL_DOWN:
			if((x+y)%2 == 0 && param == 0 /*Red step*/)
				{ 
					result = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ;
				       // storing the result in the current pixel Optimization
				       u[idx]=result; 
				}
			if((x+y)%2 == 1 && param == 1 /*Black step*/)
				{   
					result = 0.33*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+ v_r[idx]+v_l[idx]+v_u[idx]+v_d[idx]);
					solution[idx] =  result + theta*(result - solution[idx]) ; 
				       // storing the result in the current pixel Optimization
				       u[idx]=result;    
				}
		    break; 
        	   }
  
	      }
  	 }
}

//SOR redblack implementation using Shared memory- GPU
__global__ void poisson_shared_sor_redblack(int *boundary, float *u, float *solution, int w , int h, int target_nc, int boundBoxMinX, int boundBoxMinY, int boundBoxMaxX, int boundBoxMaxY, float *v_l, float *v_r, float *v_d, float *v_u, int param)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = blockIdx.z; //Using z-dimension for channels instead of for loop

    float theta = THETA; //Used in SOR
    float result = 0;    //Using local variable to minimize global memory access for optimization

    extern __shared__ float share[];

    if(x<w && y<h)
	   {
   	  //Code to populate the shared memory
	  int share_height = (2 + blockDim.y); //Square shared memory block
	  int share_width = (2 + blockDim.x); //Square shared memory block
	  
	  int bw = blockDim.x; //block_width
	  int bx = blockDim.x * blockIdx.x; //block index for starting point of each block in x-dimension
	  int by = blockDim.y * blockIdx.y; //block index for starting point of each block in y-dimension
 

	  int ind = threadIdx.x + bw * threadIdx.y; //Thread index inside each block

	  int  sidx_y, sidx_x;  //Shared memory index variables

	 //Populating the shared memory
	 	for (int i = ind; i < share_height * share_width; i += (blockDim.x * blockDim.y))
		{
		  sidx_y = by - 1 + i / share_width;
		  sidx_x = bx - 1 + i % share_width;

		  sidx_x = max(min(sidx_x, w-1), 0);  //Clamping shared memory
		  sidx_y = max(min(sidx_y, h-1), 0);

		  share[i] = u[sidx_x + w * sidx_y + w * h * z];
		}

	 __syncthreads();

   
  //Code for SOR red-black poisson blending using the populated Shared memory

  //indexes for shared memory
	//int idx = (threadIdx.x+1) + ((threadIdx.y+1) * share_width); //idx is not used below so commenting this, it is only for understanding.
	int idx_nextX = (threadIdx.x+1+1) + ((threadIdx.y+1) * share_width);
	int idx_prevX = (threadIdx.x) + ((threadIdx.y+1) * share_width);
	int idx_nextY = (threadIdx.x+1) + ((threadIdx.y+1+1) * share_width);
	int idx_prevY = (threadIdx.x+1) + ((threadIdx.y) * share_width);

        int gid = x + w*y + w*h*z; //Global index
      

    switch(boundary[gid])
       {
       case INSIDE_MASK:

    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.25*(share[idx_nextX]+share[idx_prevX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

         	u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.25*(share[idx_nextX]+share[idx_prevX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

      case CORNER_PIXEL_0_0:
	 
    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.5*(share[idx_nextX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.5*(share[idx_nextX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

      case CORNER_PIXEL_0_H:

    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.5*(share[idx_nextX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.5*(share[idx_nextX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;       

	case CORNER_PIXEL_W_0:

    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.5*(share[idx_prevX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.5*(share[idx_prevX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

	case CORNER_PIXEL_W_H: 

    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.5*(share[idx_prevX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.5*(share[idx_prevX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

	case  EDGE_PIXEL_RIGHT:

    		if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.33*(share[idx_prevX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.33*(share[idx_prevX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

	case  EDGE_PIXEL_LEFT:

	    	if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.33*(share[idx_nextX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.33*(share[idx_nextX]+share[idx_nextY]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;      

	case  EDGE_PIXEL_UP:

	    	if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.33*(share[idx_nextX]+share[idx_prevX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

	        u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.33*(share[idx_nextX]+share[idx_prevX]+share[idx_prevY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break;

	case  EDGE_PIXEL_DOWN:

	    	if((x+y)%2 == 0 && param == 0 ) //Red step
		{ 
		result = 0.33*(share[idx_nextX]+share[idx_prevX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

	        u[gid] = result;
		}

		if((x+y)%2 == 1 && param == 1) //Black step
		{   
		result = 0.33*(share[idx_nextX]+share[idx_prevX]+share[idx_nextY]+ v_r[gid]+v_l[gid]+v_u[gid]+v_d[gid]);

	    	solution[gid] =  result + theta*(result - solution[gid]) ;

		u[gid] = result;
		}

		break; 
	       }
    }
}


//CPU Poisson simple
void poisson_cpu(int *f, float *s, float *u, float *t, int w , int h,int c,int iterations )
{
	int idx, idx_nextX,idx_nextY,idx_prevX,idx_prevY;
	for(int i=0; i<iterations; i++)
		{
	  for(int x=0; x<=w-1; x++)
	  {
		 for(int y=0; y<=h-1; y++)
			{
				idx = x + w*y +w*h*c;
				idx_nextX = x+1 + w*y +w*h*c;
				idx_prevX = x-1 + w*y + w*h*c;
				idx_nextY = x + w*(y+1) +w*h*c;
				idx_prevY = x + w*(y-1) +w*h*c;            


		//if we take guiding gradient as 0    
			/*	 if(f[idx]==0 && x<w-1 && y<h-1 && x>0 && y>0) 
					   u[idx] = 0.25*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+u[idx_prevY]);
                        */

		//if we take guiding gradient as source image gradient
				if(f[idx]==0 && x<w-1 && y<h-1 && x>0 && y>0) 
					u[idx] = 0.25*(u[idx_nextX]+u[idx_prevX]+u[idx_nextY]+u[idx_prevY]+4*s[idx]-s[idx_nextX]-s[idx_nextY]-s[idx_prevX]-s[idx_prevY]);

			}
	 }
		}
}



int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  
    CUDA_CHECK;

    int iterations=ITERATIONS;

    // input from command prompt :  source_image ,target_image and mask
    string source_image = "";
    string mask = "";

    bool ret;

    ret = getParam("it", iterations, argc, argv);
    if (!ret) cerr << "No Iteration Count passed" << endl;
    cout<<" Iteration Count   : "<<iterations<<endl;

    ret = getParam("s", source_image, argc, argv);
    if (!ret) cerr << "ERROR: no source_image specified" << endl;
    cout<<" source_image   : "<<source_image<<endl;

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << " gray: " << gray << endl;

#ifdef CAMERA
    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;

  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mTargetImage;
    camera >> mTargetImage;
#else
    string target_image = "";
    ret = getParam("t", target_image, argc, argv);
    if (!ret) cerr << "ERROR: no target_image specified" << endl;
    cout<<" target_image   : "<<target_image<<endl;
    // Load the input source_image using opencv 
    cv::Mat mTargetImage = cv::imread(target_image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
#endif

    // check for Target Image
    if (mTargetImage.data == NULL) { cerr << "ERROR: Could not load target image "<< endl; return 1; }

    ret = getParam("m", mask, argc, argv);
    if (!ret) cerr << "ERROR: no mask specified" << endl;
    cout<<" Mask name   : "<<mask <<endl;

    if (argc <= 1) { cout << "Usage: " << argv[0] << " -s <source_image> -t <target_image>  -m <mask>  [-it <iterations>] [-repeats <repeats>] [-gray]" << endl; return 1; }
   
    // Load the input source_image using opencv 
    cv::Mat mSourceImage = cv::imread(source_image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    if (mSourceImage.data == NULL) { cerr << "ERROR: Could not load source image " << source_image << endl; return 1; }

    // Load the input source_image using opencv 
    cv::Mat mmask = cv::imread(mask.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    if (mmask.data == NULL) { cerr << "ERROR: Could not load mask image " << mask << endl; return 1; }


    // convert to float representation (opencv loads image values as single bytes by default)
    mSourceImage.convertTo(mSourceImage,CV_32F);
    mTargetImage.convertTo(mTargetImage,CV_32F);
    mmask.convertTo(mmask,CV_32F);

    // convert range of each channel to [0,1] (opencv default is [0,255])
    mSourceImage /= 255.f;
    mTargetImage /= 255.f;
    mmask /= 255.f;

    // get source image dimensions
    int source_w = mSourceImage.cols;         // width
    int source_h = mSourceImage.rows;         // height
    int source_nc = mSourceImage.channels();  // number of channels
    cout <<endl<<" Source image   : " << source_w << " x " << source_h << " x " <<source_nc<<endl;

    // get target image dimensions
    int target_w = mTargetImage.cols;         // width
    int target_h = mTargetImage.rows;         // height
    int target_nc = mTargetImage.channels();  // number of channels
    cout <<endl<<" target image  : " << target_w << " x " << target_h << " x " <<target_nc<<endl;

    // get source image dimensions
    int mask_w = mmask.cols;         // width
    int mask_h = mmask.rows;         // height
    int mask_nc = mmask.channels();  // number of channels
    cout <<endl<<" mask          : " << mask_w << " x " << mask_h << " x " <<mask_nc<<endl;


    // Output Images
    cv::Mat mOutSourceImgMasked(target_h,target_w,mTargetImage.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOutShiftSourceImgMasked(target_h,target_w,mTargetImage.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOut(target_h,target_w,mTargetImage.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOutBoundryCheck(target_h,target_w,mTargetImage.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mOutNormalClone(target_h,target_w,mTargetImage.type());  // mOut will have the same number of channels as the input image, nc layers


    // allocate raw input source image array
    float *srcimgIn  = new float[(size_t)source_w*source_h*source_nc];
    // allocate raw input mask image array
    float *maskIn  = new float[(size_t)mask_w*mask_h*mask_nc];

    convert_mat_to_layered (srcimgIn, mSourceImage);
    convert_mat_to_layered (maskIn, mmask);


    // Display Source Image and Mask 
    convert_layered_to_mat(mSourceImage, srcimgIn);
    showImage("SourceImage", mSourceImage, 200, 200); 
    convert_layered_to_mat(mmask, maskIn);
    showImage("mask", mmask, 200, 200); 


    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut_Srcmask = new float[(size_t)target_w*target_h*mOutSourceImgMasked.channels()];
    float *imgOutBoundryCheck = new float[(size_t)target_w*target_h*mOut.channels()];
    int *boundryPixelArray = new int[(size_t)target_w*target_h*mOut.channels()];
    float *targetimgIn  = new float[(size_t)target_w*target_h*target_nc];
    float *imgOutNormalClone = new float[(size_t)target_w*target_h*mOutSourceImgMasked.channels()];
    float *imgOut = new float[(size_t)target_w*target_h*mOut.channels()];

    ///////////////////////// Start of GPU IMPLEMENTATION /////////////////////////////
    float *d_srcimgIn;
    float *d_mask;
    float *d_targetimgIn;
    float *d_imgOut_Srcmask;
    float *d_imgOutNormalClone;
    float *d_imgOut;
    float *d_imgOutBoundryCheck;
    int *d_boundryPixelArray;
    float *d_solution;
    float *d_v_l, *d_v_r, *d_v_d, *d_v_u; //Declaring relative gradient variables

    // Allocating memory
    cudaMalloc( &d_srcimgIn, source_w*source_h*source_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_mask, mask_w*mask_h*mask_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_imgOut_Srcmask, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_solution, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_imgOutBoundryCheck, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_boundryPixelArray, target_w*target_h*target_nc * sizeof(int) );
    CUDA_CHECK;
    cudaMalloc( &d_targetimgIn, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_imgOutNormalClone, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_imgOut, target_w*target_h*target_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_v_l, source_w*source_h*source_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_v_r, source_w*source_h*source_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_v_d, source_w*source_h*source_nc * sizeof(float) );
    CUDA_CHECK;
    cudaMalloc( &d_v_u, source_w*source_h*source_nc * sizeof(float) );
    CUDA_CHECK;

    // copying memory to GPU Source Image and Mask
    cudaMemcpy( d_srcimgIn, srcimgIn, source_w*source_h*source_nc* sizeof(float), cudaMemcpyHostToDevice ); 
    CUDA_CHECK;
    cudaMemcpy( d_mask, maskIn, mask_w*mask_h*mask_nc* sizeof(float), cudaMemcpyHostToDevice ); 
    CUDA_CHECK;

    // Processing : Launching Kernels
    
    dim3 block (32,4,1); 
    dim3 grid = dim3( (target_w+ block.x - 1) / block.x , (target_h + block.y - 1 ) / block.y ,1 );

#ifdef SHARED
    //dim variable declarations for shared memory
    dim3 block_shared (32,4,1); 
    dim3 grid_shared = dim3((target_w+ block.x - 1) / block.x , (target_h + block.y - 1) / block.y , 3 );

    size_t shared = (2 + block_shared.x) * (2 + block_shared.y) *sizeof(float);
#endif

    // Extracting desired portion of the image using Mask
    SourceImageMasking <<< grid , block >>> (d_srcimgIn ,d_mask, d_imgOut_Srcmask , source_nc, source_w, source_h);  

  
    cudaMemcpy( imgOut_Srcmask, d_imgOut_Srcmask, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
    CUDA_CHECK;
    convert_layered_to_mat(mOutSourceImgMasked, imgOut_Srcmask);
    showImage("Source_Image_Merged_with_Mask", mOutSourceImgMasked, 200, 200); 


    // Extracting Boundry Pixels
    ExtractingBoundryPixels <<< grid , block >>> (d_mask ,d_boundryPixelArray , source_nc, source_w, source_h);   
    
    //Calculating now BoundboxMinMaxXY using cpu call
    cudaDeviceSynchronize();
    CUDA_CHECK;

    cudaMemcpy(boundryPixelArray , d_boundryPixelArray, target_nc*target_w*target_h * sizeof(float), cudaMemcpyDeviceToHost ); 
    CUDA_CHECK;
    
    int boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY; //Declaring CPU boundary variables

    calculate_boundBoxMinMax(target_w, target_h, target_nc, boundryPixelArray, &boundBoxMinX, &boundBoxMinY, &boundBoxMaxX, &boundBoxMaxY);

    cudaDeviceSynchronize();
    CUDA_CHECK;

    cudaMemcpy(boundryPixelArray, d_boundryPixelArray, source_nc*source_w*source_h * sizeof(int), cudaMemcpyDeviceToHost ); 
    CUDA_CHECK;


#ifndef CPU //if not CPU version then only the below variables will be declared
    int selected_w, selected_h;     //Declaring selected region variables

    dim3 grid_selected; //declaring for launching kernel for only selected region- the boundBox approach
#endif

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(300) < 0)
    {
	    camera >> mTargetImage;
	    // convert to float representation (opencv loads image values as single bytes by default)
	    mTargetImage.convertTo(mTargetImage,CV_32F);
	    // convert range of each channel to [0,1] (opencv default is [0,255])
	    mTargetImage /= 255.f;
#endif
        convert_mat_to_layered (targetimgIn, mTargetImage);

        // Copying from Host to Device the Target Image inside while loop to allow for new dynamic target images when used with webcam
	    cudaMemcpy( d_targetimgIn, targetimgIn, target_w*target_h*target_nc* sizeof(float), cudaMemcpyHostToDevice ); 
	    CUDA_CHECK;

	// Pasting desired portion of the source image to Target Image- Normal Cloning without any blending technique
	    SourceMaskImageMergeinTargetImage <<< grid , block >>> (d_imgOut_Srcmask ,d_targetimgIn, d_imgOutNormalClone , target_nc, target_w, target_h); 
	    cudaMemcpy( imgOutNormalClone, d_imgOutNormalClone, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	    CUDA_CHECK;
        //End of Normal Clone

   	// Boundry Test- Only for Test purpose- Hence Commented
	/* BoundryTest <<< grid , block >>> (d_imgOut ,d_boundryPixelArray, d_imgOutBoundryCheck , target_nc, target_w, target_h); 
	   cudaMemcpy( imgOutBoundryCheck, d_imgOutBoundryCheck, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
           CUDA_CHECK;
	   convert_layered_to_mat(mOutBoundryCheck, imgOutBoundryCheck);
	   showImage("FinalImage_BoundryCheck", mOutBoundryCheck, 200, 200); 
        */

        //For initializing the Output image with Dirichlet Boundary conditions
	    initialize <<< grid , block >>> (d_targetimgIn, d_imgOut, d_boundryPixelArray, d_imgOut_Srcmask, target_w , target_h, target_nc);    
       CUDA_CHECK;

#ifndef CPU  //if not for CPU version then do the following steps. 
       //Call Kernel to calculate the guiding gradient. This is one-time activity and so done outside the iteration loop for optimization.
 	    evaluate_gradient <<< grid, block >>> (d_v_l, d_v_r, d_v_d, d_v_u, d_boundryPixelArray, d_imgOut_Srcmask, d_targetimgIn, target_nc, target_w, target_h); 
        CUDA_CHECK;

#ifdef SHARED
	  CopyImageoutinSolution <<< grid , block >>> (d_imgOut, d_solution, target_nc, target_w, target_h);
      CUDA_CHECK;
#endif


#ifdef SOR
       CopyImageoutinSolution <<< grid , block >>> (d_imgOut, d_solution, target_nc, target_w, target_h);
       CUDA_CHECK;
#endif

	   selected_w = boundBoxMaxX - boundBoxMinX + 1;
	   selected_h = boundBoxMaxY - boundBoxMinY + 1;

	   grid_selected = dim3((selected_w+ block.x - 1) / block.x , (selected_h + block.y - 1 ) / block.y ,1 ); //Defining new grid for only selected region- the boundBox

	   float time_gpu;
	   Timer timer_GPU; 
           timer_GPU.start();


	   for(int i=0; i<iterations; i++)
	   {   

#ifdef GAUSS
	//Gauss Siedel
   	       poisson_gauss_seidel  <<< grid_selected , block >>> (d_boundryPixelArray , d_imgOut, target_w , target_h, target_nc, boundBoxMinX, boundBoxMinY,boundBoxMaxX, boundBoxMaxY, d_v_l, d_v_r, d_v_d, d_v_u);
               CUDA_CHECK;
#endif

#ifdef SOR
  
     	//SOR Red black scheme
		poisson_sor_redblack  <<< grid_selected , block >>> (d_boundryPixelArray , d_imgOut, d_solution, target_w , target_h, target_nc, boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY, d_v_l, d_v_r, d_v_d, d_v_u, 0); //RED

		poisson_sor_redblack  <<< grid_selected , block >>> (d_boundryPixelArray , d_imgOut, d_solution, target_w , target_h, target_nc, boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY, d_v_l, d_v_r, d_v_d, d_v_u, 1); //BLACK
                
               CUDA_CHECK;
#endif

#ifdef SHARED
//SOR Red black scheme with shared memory
            	poisson_shared_sor_redblack  <<< grid_shared , block_shared, shared >>> (d_boundryPixelArray , d_imgOut, d_solution, target_w , target_h, target_nc, boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY, d_v_l, d_v_r, d_v_d, d_v_u, 0); //RED
 
            	poisson_shared_sor_redblack  <<< grid_shared , block_shared, shared >>> (d_boundryPixelArray , d_imgOut, d_solution, target_w , target_h, target_nc, boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY, d_v_l, d_v_r, d_v_d, d_v_u, 1); //BLACK

                CUDA_CHECK;
#endif

	   }

       timer_GPU.end();  
       time_gpu = timer_GPU.get();  // elapsed time in seconds
       cout << "Time GPU : " << time_gpu*1000 << " ms" << endl;
#ifdef SHARED
	//copy result back to host (CPU) memory
         cudaMemcpy( imgOut, d_solution, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 	
	     CUDA_CHECK;
#endif

#ifdef SOR 
	//copy result back to host (CPU) memory
	    cudaMemcpy( imgOut, d_solution, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	    CUDA_CHECK;
#endif

#ifdef GAUSS 
	    cudaMemcpy( imgOut, d_imgOut, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	    CUDA_CHECK;
#endif


#else  //Else part for CPU Poisson implementation
  

	cudaDeviceSynchronize();
	CUDA_CHECK;

	cudaMemcpy( boundryPixelArray, d_boundryPixelArray, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	CUDA_CHECK;

	cudaMemcpy( imgOut_Srcmask, d_imgOut_Srcmask, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	CUDA_CHECK;

	cudaMemcpy( targetimgIn, d_targetimgIn, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	CUDA_CHECK;

	cudaMemcpy( imgOut, d_imgOut, source_nc*source_w*source_h * sizeof(float), cudaMemcpyDeviceToHost ); 
	CUDA_CHECK;

	cudaDeviceSynchronize();
	CUDA_CHECK;

       float time_cpu;
       Timer timer_CPU; 
       timer_CPU.start();

		for(int c=0; c<target_nc; c++)
          {
		    poisson_cpu(boundryPixelArray , imgOut_Srcmask , imgOut, targetimgIn, target_w , target_h,c,iterations);

          }
       timer_CPU.end();  
       time_cpu = timer_CPU.get();  // elapsed time in seconds
       cout << "time CPU : " << time_cpu*1000 << " ms" << endl;

#endif //Endif for CPU Poisson implementation

	convert_layered_to_mat(mTargetImage, targetimgIn);
	showImage("targetImage", mTargetImage, 200, 200); 

	convert_layered_to_mat(mOutNormalClone, imgOutNormalClone);
	showImage("NormalClone", mOutNormalClone, 200, 200); 

	convert_layered_to_mat(mOut, imgOut);
	showImage("FinalImage", mOut, 800, 200); 

#ifdef CAMERA
	    // end of camera loop
 }

#else
    cv::imwrite("mask_input.jpg",mmask*255.f);
    // save input and result
    cv::imwrite("Sourceimage_input.jpg",mSourceImage*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("targetimage_input.jpg",mTargetImage*255.f);
    cv::imwrite("Source_Image_Merged_with_Mask.jpg",mOutSourceImgMasked*255.f);
    cv::imwrite("FinalImage.jpg",mOut*255.f);

    // wait for key inputs
    cv::waitKey(1000000);
#endif

    // free allocated arrays and GPU arrays
    cudaFree(d_targetimgIn);
    CUDA_CHECK;
    delete[] targetimgIn;

    cudaFree(d_imgOut);
    CUDA_CHECK;
    delete[] imgOut;

    cudaFree(d_imgOutNormalClone);
    CUDA_CHECK;
    delete[] imgOutNormalClone;

    cudaFree(d_srcimgIn);
	CUDA_CHECK;
    delete[] srcimgIn;

    cudaFree(d_mask);
    CUDA_CHECK;
    delete[] maskIn;

    cudaFree(d_imgOut_Srcmask);
    CUDA_CHECK;
    delete[] imgOut_Srcmask;

    cudaFree(d_imgOutBoundryCheck);
    CUDA_CHECK;
    delete[] imgOutBoundryCheck;

    cudaFree(d_boundryPixelArray);
    CUDA_CHECK;
    delete[] boundryPixelArray;

    cudaFree(d_solution);
    CUDA_CHECK;

    cudaFree(d_v_l);
    CUDA_CHECK;
    cudaFree(d_v_r);
    CUDA_CHECK;
    cudaFree(d_v_d);
    CUDA_CHECK;
    cudaFree(d_v_u);
    CUDA_CHECK;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



