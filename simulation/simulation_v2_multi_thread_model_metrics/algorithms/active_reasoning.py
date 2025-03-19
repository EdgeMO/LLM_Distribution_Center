import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

from algorithms.base_allocator import BaseAllocator

class ActiveReasoningTaskAllocator(BaseAllocator):
    def __init__(self, edge_nodes, models, network_env, feature_dim, 
                 exploration_rate=0.2, decay_rate=0.995):
        """
        初始化基于主动推理的任务分配器
        
        参数:
        - edge_nodes: 边缘节点列表
        - models: 可用模型列表
        - network_env: 网络环境
        - feature_dim: 任务特征维度
        - exploration_rate: 初始探索率
        - decay_rate: 探索率衰减系数
        """
        super().__init__(edge_nodes, models, network_env)
        self.feature_dim = feature_dim
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        
        # 确定使用的CPU核心数量
        n_jobs = -1  # 使用所有可用核心
        
        # 为每个(节点,模型)对创建预测模型
        self.node_model_predictors = {}
        for node in edge_nodes:
            for model in models:
                key = (node.node_id, model.model_id)
                self.node_model_predictors[key] = {
                    "accuracy": RandomForestRegressor(n_jobs=n_jobs),  # 并行训练
                    "latency": RandomForestRegressor(n_jobs=n_jobs),   # 并行训练
                    "resource": RandomForestRegressor(n_jobs=n_jobs),  # 并行训练
                    "features": [],
                    "accuracy_history": [],
                    "latency_history": [],
                    "resource_history": [],
                    "is_fitted": False  # 添加标志以跟踪模型是否已训练
                }
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self.scaler_fitted = False  # 添加标志以跟踪缩放器是否已训练
        
        # 最近使用的模型缓存 - 用于LRU替换策略
        self.model_usage_time = {}
    
    def _train_node_model_predictors(self):
        """训练节点-模型性能预测器"""
        # 每10次迭代才训练一次模型，减少计算开销
        if hasattr(self, 'train_counter'):
            self.train_counter += 1
        else:
            self.train_counter = 0
        
        if self.train_counter % 10 != 0:
            return
        
        # 收集所有特征数据
        all_features = []
        for predictor in self.node_model_predictors.values():
            all_features.extend(predictor["features"])
        
        # 如果没有足够的数据，跳过训练
        if len(all_features) < 10:
            return
        
        # 特征缩放
        self.scaler.fit(all_features)
        self.scaler_fitted = True
        
        # 训练每个预测器
        for key, predictor in self.node_model_predictors.items():
            if len(predictor["features"]) > 5:  # 至少需要5个样本
                try:
                    scaled_features = self.scaler.transform(predictor["features"])
                    
                    # 训练准确率模型
                    predictor["accuracy"].fit(scaled_features, predictor["accuracy_history"])
                    
                    # 训练延迟模型
                    predictor["latency"].fit(scaled_features, predictor["latency_history"])
                    
                    # 训练资源使用模型
                    predictor["resource"].fit(scaled_features, predictor["resource_history"])
                    
                    # 标记模型已训练
                    predictor["is_fitted"] = True
                except Exception as e:
                    print(f"训练预测器时出错: {e}")
                    predictor["is_fitted"] = False
    
    def _predict_performance(self, node_id, model_id, task_features):
        """预测任务在指定节点和模型下的性能"""
        key = (node_id, model_id)
        predictor = self.node_model_predictors.get(key)
        
        if predictor is None:
            # 如果没有这个节点-模型对的预测器，返回默认值
            return {
                "accuracy": 0.5,
                "latency": 100,
                "resource": 0.5
            }
        
        # 如果历史数据不足或模型未训练，使用模型默认值
        if len(predictor["features"]) < 5 or not predictor["is_fitted"]:
            model = next((m for m in self.models if m.model_id == model_id), None)
            if model:
                return {
                    "accuracy": model.accuracy,
                    "latency": model.inference_time,
                    "resource": model.resource_usage
                }
            else:
                return {
                    "accuracy": 0.5,
                    "latency": 100,
                    "resource": 0.5
                }
        
        try:
            # 缩放特征
            if not self.scaler_fitted:
                # 如果缩放器未训练，返回默认值
                model = next((m for m in self.models if m.model_id == model_id), None)
                if model:
                    return {
                        "accuracy": model.accuracy,
                        "latency": model.inference_time,
                        "resource": model.resource_usage
                    }
                else:
                    return {
                        "accuracy": 0.5,
                        "latency": 100,
                        "resource": 0.5
                    }
            
            scaled_features = self.scaler.transform([task_features])[0].reshape(1, -1)
            
            # 预测性能
            accuracy = predictor["accuracy"].predict(scaled_features)[0]
            latency = predictor["latency"].predict(scaled_features)[0]
            resource = predictor["resource"].predict(scaled_features)[0]
            
            return {
                "accuracy": accuracy,
                "latency": latency,
                "resource": resource
            }
        except Exception as e:
            print(f"预测性能时出错: {e}")
            # 发生错误时返回默认值
            model = next((m for m in self.models if m.model_id == model_id), None)
            if model:
                return {
                    "accuracy": model.accuracy,
                    "latency": model.inference_time,
                    "resource": model.resource_usage
                }
            else:
                return {
                    "accuracy": 0.5,
                    "latency": 100,
                    "resource": 0.5
                }
    
    def _deploy_model(self, model_id, node_id):
        """将模型部署到节点上 (重写基类方法)"""
        # 获取模型和节点对象
        model = next((m for m in self.models if m.model_id == model_id), None)
        node = next((n for n in self.edge_nodes if n.node_id == node_id), None)
        
        if model is None or node is None:
            return False
        
        # 检查模型是否已部署
        if model_id in self.model_deployment_cache.get(node_id, set()):
            # 更新最近使用时间
            self.model_usage_time[(node_id, model_id)] = time.time()
            return True
        
        # 尝试部署模型
        if node.can_deploy_model(model):
            success = node.deploy_model(model)
            if success:
                # 更新缓存
                self.model_deployment_cache[node_id].add(model_id)
                self.model_usage_time[(node_id, model_id)] = time.time()
                return True
        
        # 如果无法直接部署，尝试LRU替换
        if len(node.deployed_models) > 0:
            # 找出最近最少使用的模型
            lru_candidates = []
            for deployed_model_id in node.deployed_models:
                usage_time = self.model_usage_time.get((node_id, deployed_model_id), 0)
                lru_candidates.append((usage_time, deployed_model_id))
            
            # 按使用时间排序
            lru_candidates.sort()
            
            # 移除最久未使用的模型，直到有足够空间
            for _, old_model_id in lru_candidates:
                # 确保模型存在于缓存中再移除
                if old_model_id in self.model_deployment_cache.get(node_id, set()):
                    node.remove_model(old_model_id)
                    self.model_deployment_cache[node_id].remove(old_model_id)
                    
                    # 尝试部署新模型
                    if node.can_deploy_model(model):
                        success = node.deploy_model(model)
                        if success:
                            self.model_deployment_cache[node_id].add(model_id)
                            self.model_usage_time[(node_id, model_id)] = time.time()
                            return True
        
        return False
    
    def allocate_tasks(self, task_batch, current_bandwidth):
        """分配任务到边缘节点并选择模型"""
        # 更新网络环境带宽
        for node in self.edge_nodes:
            self.network_env.set_bandwidth(node.node_id, current_bandwidth)
        
        # 定期训练预测器
        self._train_node_model_predictors()
        
        allocations = []
        
        for task_features in task_batch:
            # 增加总任务计数
            self.total_tasks += 1
            
            # 探索 vs 利用
            if np.random.random() < self.exploration_rate:
                # 随机选择节点和模型
                node = np.random.choice(self.edge_nodes)
                model = np.random.choice(self.models)
                
                # 尝试部署模型
                transfer_time = self._calculate_model_transfer_time(model.model_id, node.node_id)
                deploy_success = self._deploy_model(model.model_id, node.node_id)
                
                # 记录模型选择
                if model.model_id not in self.model_selection_counts:
                    self.model_selection_counts[model.model_id] = 0
                self.model_selection_counts[model.model_id] += 1
                
                # 记录传输时间
                self.transfer_times.append(transfer_time)
                
                # 记录卸载收益（如果部署了新模型）
                if transfer_time > 0:
                    accuracy_before = self._get_default_accuracy(task_features)
                    accuracy_after = model.accuracy
                    accuracy_gain = (accuracy_after - accuracy_before) * 100  # 转换为百分比
                    transfer_cost = transfer_time * 1000  # 转换为毫秒
                    
                    self.model_offloading_benefits.append({
                        "accuracy_gain": accuracy_gain,
                        "transfer_cost": transfer_cost
                    })
                    
                    # 记录传输延迟
                    self.transfer_latencies.append(transfer_time * 1000)  # 转换为毫秒
                else:
                    # 记录传输延迟为0
                    self.transfer_latencies.append(0)
                
                # 记录推理延迟（预估）
                self.inference_latencies.append(model.inference_time)
                
                if deploy_success:
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node.node_id,
                        "model_id": model.model_id,
                        "transfer_time": transfer_time
                    })
                else:
                    # 如果部署失败，选择已部署的模型
                    deployed_models = list(self.model_deployment_cache.get(node.node_id, set()))
                    if deployed_models:
                        model_id = np.random.choice(deployed_models)
                        
                        # 更新模型选择计数
                        if model_id not in self.model_selection_counts:
                            self.model_selection_counts[model_id] = 0
                        self.model_selection_counts[model_id] += 1
                        
                        # 使用已部署模型的推理延迟
                        selected_model = next((m for m in self.models if m.model_id == model_id), None)
                        if selected_model:
                            self.inference_latencies.append(selected_model.inference_time)
                        
                        allocations.append({
                            "task_features": task_features,
                            "node_id": node.node_id,
                            "model_id": model_id,
                            "transfer_time": 0  # 已部署，无需传输
                        })
                    else:
                        # 如果节点没有模型，选择最小的模型
                        smallest_model = min(self.models, key=lambda m: m.size_mb)
                        transfer_time = self._calculate_model_transfer_time(smallest_model.model_id, node.node_id)
                        self._deploy_model(smallest_model.model_id, node.node_id)
                        
                        # 更新模型选择计数
                        if smallest_model.model_id not in self.model_selection_counts:
                            self.model_selection_counts[smallest_model.model_id] = 0
                        self.model_selection_counts[smallest_model.model_id] += 1
                        
                        # 记录传输时间
                        self.transfer_times.append(transfer_time)
                        
                        # 记录延迟
                        self.transfer_latencies.append(transfer_time * 1000)  # 转换为毫秒
                        self.inference_latencies.append(smallest_model.inference_time)
                        
                        allocations.append({
                            "task_features": task_features,
                            "node_id": node.node_id,
                            "model_id": smallest_model.model_id,
                            "transfer_time": transfer_time
                        })
            else:
                # 利用：为每个(节点,模型)对计算得分
                best_score = float('-inf')
                best_allocation = None
                
                for node in self.edge_nodes:
                    for model in self.models:
                        # 预测性能
                        performance = self._predict_performance(node.node_id, model.model_id, task_features)
                        
                        # 计算模型传输时间
                        transfer_time = self._calculate_model_transfer_time(model.model_id, node.node_id)
                        
                        # 计算总延迟 (传输时间 + 推理时间)
                        total_latency = transfer_time * 1000 + performance["latency"]  # 转换为ms
                        
                        # 计算综合得分
                        # 高准确率、低延迟、低资源使用为佳
                        score = (performance["accuracy"] * 10 - 
                                total_latency / 100 - 
                                performance["resource"] * 0.5)
                        
                        if score > best_score:
                            best_score = score
                            best_allocation = {
                                "task_features": task_features,
                                "node_id": node.node_id,
                                "model_id": model.model_id,
                                "transfer_time": transfer_time,
                                "predicted_performance": performance
                            }
                
                # 尝试部署最佳模型
                if best_allocation:
                    deploy_success = self._deploy_model(best_allocation["model_id"], best_allocation["node_id"])
                    
                    # 记录模型选择
                    if best_allocation["model_id"] not in self.model_selection_counts:
                        self.model_selection_counts[best_allocation["model_id"]] = 0
                    self.model_selection_counts[best_allocation["model_id"]] += 1
                    
                    # 记录传输时间
                    self.transfer_times.append(best_allocation["transfer_time"])
                    
                    # 记录延迟构成
                    self.transfer_latencies.append(best_allocation["transfer_time"] * 1000)  # 转换为毫秒
                    self.inference_latencies.append(best_allocation["predicted_performance"]["latency"])
                    
                    # 记录卸载收益（如果部署了新模型）
                    if best_allocation["transfer_time"] > 0:
                        accuracy_before = self._get_default_accuracy(task_features)
                        accuracy_after = best_allocation["predicted_performance"]["accuracy"]
                        accuracy_gain = (accuracy_after - accuracy_before) * 100  # 转换为百分比
                        transfer_cost = best_allocation["transfer_time"] * 1000  # 转换为毫秒
                        
                        self.model_offloading_benefits.append({
                            "accuracy_gain": accuracy_gain,
                            "transfer_cost": transfer_cost
                        })
                    
                    if deploy_success:
                        allocations.append({
                            "task_features": task_features,
                            "node_id": best_allocation["node_id"],
                            "model_id": best_allocation["model_id"],
                            "transfer_time": best_allocation["transfer_time"]
                        })
                    else:
                        # 如果部署失败，在已部署的模型中选择最佳的
                        node_id = best_allocation["node_id"]
                        deployed_models = list(self.model_deployment_cache.get(node_id, set()))
                        
                        if deployed_models:
                            best_deployed_score = float('-inf')
                            best_deployed_model = None
                            
                            for model_id in deployed_models:
                                performance = self._predict_performance(node_id, model_id, task_features)
                                score = (performance["accuracy"] * 10 - 
                                        performance["latency"] / 100 - 
                                        performance["resource"] * 0.5)
                                
                                if score > best_deployed_score:
                                    best_deployed_score = score
                                    best_deployed_model = model_id
                            
                            if best_deployed_model:
                                # 更新模型选择计数
                                if best_deployed_model not in self.model_selection_counts:
                                    self.model_selection_counts[best_deployed_model] = 0
                                self.model_selection_counts[best_deployed_model] += 1
                                
                                # 记录推理延迟
                                performance = self._predict_performance(node_id, best_deployed_model, task_features)
                                self.inference_latencies.append(performance["latency"])
                                self.transfer_latencies.append(0)  # 已部署，无传输延迟
                                
                                allocations.append({
                                    "task_features": task_features,
                                    "node_id": node_id,
                                    "model_id": best_deployed_model,
                                    "transfer_time": 0  # 已部署，无需传输
                                })
                                continue
                        
                        # 如果节点没有模型，选择其他节点
                        other_nodes = [n for n in self.edge_nodes if n.node_id != node_id]
                        if other_nodes:
                            node = np.random.choice(other_nodes)
                            smallest_model = min(self.models, key=lambda m: m.size_mb)
                            transfer_time = self._calculate_model_transfer_time(smallest_model.model_id, node.node_id)
                            self._deploy_model(smallest_model.model_id, node.node_id)
                            
                            # 更新模型选择计数
                            if smallest_model.model_id not in self.model_selection_counts:
                                self.model_selection_counts[smallest_model.model_id] = 0
                            self.model_selection_counts[smallest_model.model_id] += 1
                            
                            # 记录传输时间和延迟
                            self.transfer_times.append(transfer_time)
                            self.transfer_latencies.append(transfer_time * 1000)  # 转换为毫秒
                            self.inference_latencies.append(smallest_model.inference_time)
                            
                            allocations.append({
                                "task_features": task_features,
                                "node_id": node.node_id,
                                "model_id": smallest_model.model_id,
                                "transfer_time": transfer_time
                            })
        
        # 衰减探索率
        self.exploration_rate *= self.decay_rate
        
        # 如果没有部署记录，添加0
        if len(self.deployment_history) == 0 or self.deployment_history[-1] != 0:
            self.deployment_history.append(0)
        
        return allocations
    def update_feedback(self, executed_allocations):
        """更新来自边缘节点的反馈"""
        total_accuracy = 0
        total_latency = 0
        total_resource = 0
        count = 0
        
        for alloc in executed_allocations:
            task_features = alloc["task_features"]
            node_id = alloc["node_id"]
            model_id = alloc["model_id"]
            transfer_time = alloc["transfer_time"]
            accuracy = alloc["result"]["accuracy"]
            latency = alloc["result"]["latency"]
            resource = alloc["result"]["resource_usage"]
            
            # 考虑传输时间
            total_latency_with_transfer = latency + transfer_time * 1000  # 转换为ms
            
            # 更新预测器的历史数据
            key = (node_id, model_id)
            if key in self.node_model_predictors:
                predictor = self.node_model_predictors[key]
                predictor["features"].append(task_features)
                predictor["accuracy_history"].append(accuracy)
                predictor["latency_history"].append(total_latency_with_transfer)
                predictor["resource_history"].append(resource)
            
            # 累计性能指标
            total_accuracy += accuracy
            total_latency += total_latency_with_transfer
            total_resource += resource
            count += 1
        
        if count > 0:
            avg_accuracy = total_accuracy / count
            avg_latency = total_latency / count
            avg_resource = total_resource / count
            
            # 计算综合性能得分
            performance = avg_accuracy - 0.01 * avg_latency + 0.5 * avg_resource
            self.performance_history.append(performance)
            
            # 检查收敛性
            if len(self.performance_history) > 20 and not self.converged:
                recent_perf = self.performance_history[-10:]
                if np.std(recent_perf) < 0.01:  # 如果性能稳定，认为已收敛
                    self.converged = True
                    self.convergence_time = time.time() - self.start_time