<template>
    <div class="app-container">
        <Header />
        <div class="chat-container">
            <div class="messages-container">
                <ChatBubble v-for="(item, index) in messagesAndAnswers" :key="index" :message="item.message"
                    :answer="item.answer" :run_id="item.run_id" />
            </div>

            <div class="input-container">
                <form @submit.prevent="submitForm" class="input-form">
                    <input type="text" v-model="message" placeholder="질문을 입력하세요" required class="input-field"
                        :disabled="isLoading" />
                    <div class="button-container">
                        <!-- <button type="submit" class="submit-button" :disabled="isLoading">Submit</button> -->
                        <button type="button" @click="resetStore" class="reset-button" :disabled="isLoading">
                            Reset
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import ChatBubble from '../components/ChatBubble.vue';

const messagesAndAnswers = ref([]);
const message = ref('');
const isLoading = ref(false);  // 로딩 상태를 나타내는 플래그 변수

const submitForm = () => {
    if (isLoading.value) return;  // 이미 로딩 중이면 함수 종료

    isLoading.value = true;  // 로딩 상태 시작

    const formData = new FormData();
    formData.append('message', message.value);

    // 메시지를 즉시 추가
    messagesAndAnswers.value.push({
        message: message.value,
        answer: null  // 초기 답변은 null로 설정
    });

    fetch(`${import.meta.env.VITE_API_URL}/`, {
        method: "POST",
        credentials: 'include', // 클라이언트와 서버가 통신할때 쿠키와 같은 인증 정보 값을 공유하겠다는 설정
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            // 응답 데이터로 메시지의 답변 업데이트
            const index = messagesAndAnswers.value.findIndex(item => item.message === message.value);
            if (index !== -1) {
                messagesAndAnswers.value[index].answer = data.answer;
                messagesAndAnswers.value[index].run_id = data.run_id;
            }
            message.value = '';  // 입력 필드 초기화
        })
        .catch(error => {
            console.error("Error:", error);
        })
        .finally(() => {
            isLoading.value = false;  // 로딩 상태 종료
        });
};

const resetStore = () => {
    if (isLoading.value) return;  // 로딩 중이면 리셋 동작 방지

    fetch(`${import.meta.env.VITE_API_URL}/reset`, {
        method: "POST",
        credentials: 'include', // 클라이언트와 서버가 통신할때 쿠키와 같은 인증 정보 값을 공유하겠다는 설정
    })
        .then(() => {
            messagesAndAnswers.value = [];  // 클라이언트 측의 messagesAndAnswers 배열 초기화
            message.value = '';  // 입력 필드 초기화
        })
        .catch(error => {
            console.error("Error:", error);
        });
};

onMounted(() => {
    resetStore();  // 페이지 로드 시 서버에서 초기 상태를 가져옴
});
</script>

<style>
@import '../assets/css/main.css';
</style>