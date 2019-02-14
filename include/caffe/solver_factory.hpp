/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {		//Solver 登记仓库
 public:

  //函数指针，输入参数为 SolverParameter& 类型的引用，返回值为一个Solver类型的指针
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);

  //键值对为 (字符串,函数指针) 的 map
  typedef std::map<string, Creator> CreatorRegistry;

  //静态成员方法,返回指向 CreatorRegistry(map) 的 static指针(第二次进来 直接返回已经存在的,不会重复 new)
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  //向 CreatorRegistry(map)添加一个 (type,creator) pair.
  static void AddCreator(const string& type, Creator creator) {
    //获取 CreatorRegistry(map) 的 static指针
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
	//添加 (字符串,函数指针) 键值对
    registry[type] = creator;
  }

  //根据 SolverParameter 从 CreatorRegistry(map) 获取一个Creator,并调用 Creator 函数得到 solver
  //static 方法只可以操作本类的 static 成员,无法操作属于具体某个对象的数据成员
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    const string& type = param.type();
    //获取 CreatorRegistry(map) 的 static指针
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
	//registry[type]返回从map拿到的 Creator 类型的函数指针,然后以 param 作为参数调用之
    return registry[type](param);
  }

  //向一个vector<string>变量中迭代插入求解器的名字，也就是type
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin(); iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);	//这是吧所有的 key 塞入 vector 了啊
    }
    return solver_types;	//返回到上层的时候,这里会发生复制(看成 int 简单数据类型即可)
  }

 private:

  //Solver registry should never be instantiated - everything is done with its static variables.
  //将类的构造函数放在private里,可以避免被实例化
  SolverRegistry() {}

  //输出所有 求解器的名字，也就是type
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();	//这里 SolverTypeList() 就是 返回了函数中 local的 vector<string> 变量的一份拷贝
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin(); iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type, Solver<Dtype>* (*creator)(const SolverParameter&)) {
    //LOG(INFO) << "Registering solver type: " << type;
    //构造函数里面直接调用 AddCreator 向 map 中添加 (字符串,函数指针) 键值对,告诉工厂怎么 create 自己啊...
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

//fang: ##字面字符串连接, #引用字面字符串
//type 以 SGD(随机梯度下降法) 为例: REGISTER_SOLVER_CREATOR(SGD, creator) 相当于
//static SolverRegisterer<float> g_creator_f_SGD(SGD, creator<float>);
//static SolverRegisterer<double> g_creator_d_SGD(SGD, creator<double>);
//定义 static 变量的时候自动调用构造函数,将 (字符串,函数指针) 键值对添加到登记仓库
//(登记时主要 AddCreator-->见 sgd_solver.cpp 等, 以后就主要是 CreateSolver 了-->见 caffe.cpp-->train().)
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \


//定义 Creator_SGDSolver() 函数,并调用上面的 REGISTER_SOLVER_CREATOR(type, creator)
//这也规定了 SGD 对应的 solver 类名字必须要是 SGDSolver,并且必须要有一个
//SGDSolver(SolverParameter param)的构造函数....
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
