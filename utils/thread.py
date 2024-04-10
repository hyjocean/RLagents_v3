import threading

# 获取当前活动的线程列表
active_threads = threading.enumerate()

for thread in active_threads:
    print(thread.name)
