<template>
    <div class="app-container">
        <Header />
        <div class="chat-container">
            <div class="messages-container" ref="messagesContainer">
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
import { ref, onMounted, watch, nextTick } from 'vue';
import ChatBubble from '../components/ChatBubble.vue';

const messagesAndAnswers = ref([]);
const message = ref('');
const isLoading = ref(false);  // 로딩 상태를 나타내는 플래그 변수
const messagesContainer = ref(null);  // messages-container 요소에 대한 참조

const config = useRuntimeConfig()
const baseURL = config.public.apiUrl

// 스크롤을 맨 아래로 이동시키는 함수
const scrollToBottom = () => {
    if (messagesContainer.value) {
        const lastMessage = messagesContainer.value.lastElementChild;
        if (lastMessage) {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
};

// messagesAndAnswers가 변경될 때마다 스크롤을 맨 아래로 이동
watch(messagesAndAnswers, () => {
    // DOM 업데이트 후 스크롤 이동을 위해 nextTick 사용
    nextTick(() => {
        scrollToBottom();
    });
}, { deep: true });

const submitForm = () => {
    if (isLoading.value) return;  // 이미 로딩 중이면 함수 종료

    isLoading.value = true;  // 로딩 상태 시작

    // 메시지를 즉시 추가
    messagesAndAnswers.value.push({
        message: message.value,
        answer: null  // 초기 답변은 null로 설정
    });

    // 입력창 비우기
    const currentMessage = message.value;
    message.value = '';

    // fetch(`${baseURL}/`, {
    fetch(`${import.meta.env.VITE_API_URL}/`, {
        method: "POST",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: currentMessage
        }),
    })
        .then(response => response.json())
        .then(data => {
            // 응답 데이터로 메시지의 답변 업데이트
            const index = messagesAndAnswers.value.findIndex(item => item.message === currentMessage);
            if (index !== -1) {
                messagesAndAnswers.value[index].answer = data.answer;
                messagesAndAnswers.value[index].run_id = data.run_id;
            }
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

    // fetch(`${baseURL}/reset`, {
    fetch(`${import.meta.env.VITE_API_URL}/reset`, {
    // fetch('http://localhost:8000/reset', {
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