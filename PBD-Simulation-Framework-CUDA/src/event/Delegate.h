#ifndef DELEGATE_H
#define DELEGATE_H

#include <utility>


template<typename T> 
class Delegate;

template<class Ret, class... Args>
class Delegate<Ret(Args...)> {

  using Stub_Pointer_Type = Ret (*)(void* object_pointer, Args&&... args);

  Delegate(void* const object_pointer, Stub_Pointer_Type const stub_pointer) 
  : object_pointer_{object_pointer}, stub_pointer_{stub_pointer} {}

public:
  Delegate() = default;
  Delegate(const Delegate& other) = default;
  //Delegate(Delegate&& other) = default; 

  ~Delegate() = default;

  template<Ret(* const function_pointer)(Args...)> 
  static Delegate from() { 
    return Delegate<Ret(Args...)>{ 
                                  nullptr, 
                                  function_stub<function_pointer>
                                 };
  }

  template<class Object, Ret(Object::* const member_function_pointer)(Args...)> 
  static Delegate from(Object* const object_pointer) {
    return Delegate<Ret(Args...)>{ 
                                  object_pointer, 
                                  member_function_stub<Object, member_function_pointer>
                                 };
  }

  template<class Object, Ret(Object::* const member_function_pointer)(Args...)> 
  static Delegate from(Object const* const object_pointer) {
    return Delegate<Ret(Args...)>{
                                  const_cast<Object*>(object_pointer), 
                                  const_member_function_stub<Object, member_function_pointer>
                                 };
  }

  Ret operator()(Args&&... args) const {
    return stub_pointer_(object_pointer_, std::forward<Args>(args)...);
  }

  bool operator==(const Delegate<Ret(Args...)>& other) const
  {
    return (object_pointer_ == other.object_pointer_) && (stub_pointer_ == other.stub_pointer_);
  }

private:
  void*               object_pointer_;
  Stub_Pointer_Type   stub_pointer_;

  template<Ret(*function_pointer)(Args...)> 
  static Ret function_stub(void* const object_pointer, Args&&... args) {
    return function_pointer(std::forward<Args>(args)...);
  }

  template<class Object, Ret(Object::*member_function_pointer)(Args...)>
  static Ret member_function_stub(void* const object_pointer, Args&&... args) {
    return (static_cast<Object*>(object_pointer)->*member_function_pointer)(std::forward<Args>(args)...);
  }

  template<class Object, Ret(Object::*member_function_pointer)(Args...) const>
  static Ret const_member_function_stub(void* const object_pointer, Args&&... args) {
    return (static_cast<const Object*>(object_pointer)->*member_function_pointer)(std::forward<Args>(args)...);
  }

};


#endif // DELEGATE_H