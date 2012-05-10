#include <iostream>
class B
{
  public:
    void print() const {
      const_cast<int&>(i) = 0; //使用const_cast<int &>(i)转换后可以在const函数中进行修改，这里是转换为引用
      *(const_cast<int *>(&i)) = 100; //使用const_cast<int *>(&i),这里是转换指针，然后通过指针进行访问
      j++;//mutable修饰的变量可以在const函数中进行修改
      std::cout << i << endl ;
  }

  private:
    int i;
    mutable int j; //使用mutable关键字修饰的变量可以在const函数中进行修改
};

void main()
{
  B b;
  b.print();
}
