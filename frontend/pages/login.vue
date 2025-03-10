<template>
    <div class="background-wrapper">
        <div class="background">
            <div class="login-form">
                <form @submit.prevent="handleSubmit" class="flex flex-col items-center">
                    <div class="mb-10 text-center">
                        <p class="font-medium text-3xl">Address Chatbot</p>
                        <p class="mt-5">Login to access the app</p>
                    </div>
                    <div class="mb-5 w-full">
                        <label for="id" class="block mb-2 text-sm font-medium text-gray-900">Your ID</label>
                        <input type="text" id="id" v-model="id"
                            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
                            placeholder="" required />
                    </div>
                    <div class="mb-5 w-full">
                        <label for="password" class="block mb-2 text-sm font-medium text-gray-900">Your PW</label>
                        <input type="password" id="password" v-model="password"
                            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
                            required />
                    </div>
                    <div class="w-full flex justify-center">
                        <button type="submit"
                            class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center">Submit</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from '#app'

const id = ref('')
const password = ref('')
const router = useRouter()

onMounted(() => {
    const storedId = localStorage.getItem('userId')
    if (storedId) {
        id.value = storedId
        router.push('/')
    }
})

const handleSubmit = async () => {
    try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}/set_user_id`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ id: id.value })
        })

        if (response.ok) {
            // 로그인 정보 localStorage에 저장
            localStorage.setItem('userId', id.value)
            router.push('/')
        } else {
            alert('Failed to send ID. Please try again.')
        }
    } catch (error) {
        console.error('Login error:', error)
        alert('An error occurred while sending ID. Please try again.')
    }
}
</script>

<style>
.background-wrapper {
    min-height: 100vh;
    /* 페이지의 최소 높이를 화면 전체로 설정 */
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f9f9f9;
}

.background {
    background-color: #f9f9f9;
    width: 100%;
    /* 부모 컨테이너의 너비에 맞춤 */
    display: flex;
    justify-content: center;
    align-items: center;
}

.login-form {
    background-color: #fff;
    border: 1px solid #eeeeee;
    border-radius: 8px;
    padding: 30px;
    width: 100%;
    max-width: 400px;
}
</style>
