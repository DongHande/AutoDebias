#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<map>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;
namespace py = pybind11;


map<long long,int>mp;
double eps=1e-15;
py::array_t<double> find_2d(py::array_t<long long>& input1, py::array_t<long long>& input2) {

    mp.clear();
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    auto p1 = input1.unchecked<2>();
    auto p2 = input2.unchecked<2>();

    if (buf1.ndim !=2 || buf2.ndim !=2)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }
   
    int nn1=buf1.shape[0];
    int nn2=buf2.shape[0];
   
   // cout<<nn1<<' '<<nn2<<' '<<dd1<<' '<<dd2<<endl;
     if (buf1.shape[1] !=2 || buf2.shape[1] !=2)
    {
        throw std::runtime_error("Number of second dimensions must be 2");
    }
    
   // cout<<nn1<<' '<<nn2<<endl;

    auto result = py::array_t<double>(buf2.size/2);
  //  cout<<buf1.size<<'g'<<buf2.size<<endl;
    py::buffer_info buf3 = result.request();
   //  cout<<nn1<<' '<<nn2<<endl;
    result.resize({nn2});
  //   cout<<nn1<<' '<<nn2<<endl;

    auto p3=result.mutable_unchecked<1>();
  //   cout<<nn1<<' '<<nn2<<endl;
    
    for (int i = 0; i < nn1; i++)
    {
        long long x=p1(i,0);
        long long y=p1(i,1);
        long long code=(x<<30)+y;
        mp[code]=1;
    }
   
 //   cout<<nn1<<' '<<nn2<<endl;
    for(int i=0;i < nn2; i++)
    {
        long long x=p2(i,0);
        long long y=p2(i,1);
        long long code=(x<<30)+y;
        if(mp.find(code)==mp.end())
        p3(i)=0;
        else
        p3(i)=1;   
     
    }
    

    return result;

}

map<long long,int>mt;
py::array_t<double> gaotest(py::array_t<long long>& input1, py::array_t<long long>& input2,py::array_t<long long>& input3, py::array_t<long long>& input4) {
    mt.clear();
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();
    py::buffer_info buf4 = input4.request();
    
    auto testu = input1.unchecked<1>();
     auto testi = input2.unchecked<1>();
    auto reitem = input3.unchecked<2>();
     auto alitem = input4.unchecked<2>();   
        
    int nn1=buf1.shape[0];
    int nn2=buf2.shape[0];
    int n=buf3.shape[0];
    int n1=buf4.shape[0];
    int ch=buf3.shape[1];
    int m=buf4.shape[1];
   
 //   cout<<nn1<<' '<<nn2<<' '<<dd1<<' '<<dd2<<endl;
     if (n!=n1||nn1!=nn2)
    {
        throw std::runtime_error("error input 3 and 4");
    }
   
     
   
    /*
     for(int i=0;i<100;i++)
    cout<<testu[i]<<'\t';
    cout<<endl;
    */
    
   // cout<<nn1<<' '<<nn2<<' '<<n<<' '<<ch<<' '<<m<<endl;  
    
    vector<double> idcg(n),logsum(m+1),pre(n),re(n),ndcg(n),mrr(n),nd5(n),id5(n);
    vector<int> num(n);
    
    for(int i=0;i<n;i++)
    num[i]=0;
    
    logsum[0]=0;
    for(int j=1;j<=m;j++)
    {
       logsum[j]=logsum[j-1]+1.0/log2(j+1);
    }
    
    for(int i=0;i<nn1;i++)
    {
        long long x=testu(i);
        if(x<0||x>=n)
            throw std::runtime_error("error x");
        num[x]++;
        long long y=testi(i);
        long long code=(x<<30)+y;
        mt[code]=1;
    }
   for(int i=0;i<n;i++)
   {
       idcg[i]=logsum[num[i]];
       id5[i]=logsum[min(num[i],ch)];
   }
   
    
 // cout<<nn1<<' '<<nn2<<' '<<n<<' '<<ch<<' '<<m<<endl;  
    int cao=0;
    
    for(long long i=0;i<n;i++)
    {
       // cout<<i<<endl;
        if(num[i]==0)
        {
            pre[i]=re[i]=ndcg[i]=mrr[i]=nd5[i]=0;
            continue;
        }
        cao++;
        double hit=0;
        double ndie=0;
        for(int j=0;j<ch;j++)
        {
          //  cout<<j<<endl;
            int item=reitem(i,j);
         //   if(i==3)
         //   cout<<j<<' '<<item<<endl;
            long long code=(i<<30)+item;
            if(mt.find(code)!=mt.end())
            {
                hit++;
                ndie+=1.0/log2(j+2);
            }
        }
       pre[i]=hit/ch;
       re[i]=hit/num[i];
      
        double ng=0;
        double mr=0;
        
        for(int j=0;j<m;j++)
        {
           int item=alitem(i,j);
           long long code=(i<<30)+item;
           if(mt.find(code)==mt.end())
           continue;
           ng+=1.0/log2(j+2);
           mr+=1.0/(j+1);
        }
        
       
       ndcg[i]=ng/idcg[i];
       mrr[i]=mr;
       nd5[i]=ndie/id5[i];

     //  if(i<=3)
    //    cout<<pre[i]<<' '<<re[i]<<' '<<ndcg[i]<<' '<<mrr[i]<<' '<<idcg[i]<<endl;
    }
   double anpre=0,anre=0,anndcg=0,anmrr=0,annd5=0;
   for(int i=0;i<n;i++)
   {
       if(num[i]!=0)
       {
           anpre+=pre[i];
           anre+=re[i];
           anndcg+=ndcg[i];
           anmrr+=mrr[i];
           annd5+=nd5[i];
       }
   }
 //   cout<<anpre<<' '<<anre<<' '<<anndcg<<' '<<anmrr<<' '<<cao<<endl;
   anpre/=cao;
   anre/=cao;
   anndcg/=cao;
   anmrr/=cao;
   annd5/=cao;
  //  cout<<anpre<<' '<<anre<<' '<<anndcg<<' '<<anmrr<<' '<<cao<<endl;
  
   
   auto result = py::array_t<double>(5);
   py::buffer_info buf5 = result.request();   
    double* out = (double*)buf5.ptr;  
    out[0]=anpre;
    out[1]=anre;
    out[2]=anndcg;
    out[3]=anmrr;
    out[4]=annd5;
    return result;
}
   
            
           
        
        
        
        
            
        
    
        
        
   


PYBIND11_MODULE(ex, m) {

    m.doc() = "Simple demo using numpy!";
    m.def("gaotest", &gaotest);
    m.def("find_2d", &find_2d);
}
/*
<%
setup_pybind11(cfg)
%>
*/