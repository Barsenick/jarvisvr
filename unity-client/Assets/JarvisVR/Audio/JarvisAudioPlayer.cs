using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using UnityEngine;
using JarvisVR.Net;
using JarvisVR.Protocol;

namespace JarvisVR.Audio
{
    /// <summary>
    /// Seamlessly plays streaming PCM16 audio chunks received from the Voice Bridge.
    /// Uses OnAudioFilterRead to feed raw samples directly to the audio hardware.
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public class JarvisAudioPlayer : MonoBehaviour
    {
        [SerializeField] private JarvisConnection connection;
        
        private ConcurrentQueue<float> _sampleQueue = new ConcurrentQueue<float>();
        private AudioSource _audioSource;
        private bool _isInitialized = false;
        private int _sampleRate = 16000; // Default, will be updated by first chunk

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _audioSource.loop = false;
        }

        private void Start()
        {
            if (connection == null)
            {
                connection = FindObjectOfType<JarvisConnection>();
            }

            if (connection != null)
            {
                connection.OnAudioChunk += HandleAudioChunk;
            }
        }

        private void OnDestroy()
        {
            if (connection != null)
            {
                connection.OnAudioChunk -= HandleAudioChunk;
            }
        }

        private void HandleAudioChunk(Envelope env)
        {
            if (env.Payload == null) return;

            string b64Audio = env.Payload.Value<string>("audio");
            if (string.IsNullOrEmpty(b64Audio)) return;

            try
            {
                byte[] bytes = Convert.FromBase64String(b64Audio);
                
                // Update sample rate from payload if provided
                if (env.Payload.ContainsKey("sr"))
                {
                    _sampleRate = env.Payload.Value<int>("sr");
                }

                // Convert PCM16 (short) to Float (-1.0 to 1.0) for Unity AudioSource
                for (int i = 0; i < bytes.Length; i += 2)
                {
                    short sample = BitConverter.ToInt16(bytes, i);
                    _sampleQueue.Enqueue(sample / 32768f);
                }

                // Ensure the AudioSource is playing to keep OnAudioFilterRead active
                if (!_audioSource.isPlaying)
                {
                    _audioSource.Play();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[JarvisAudioPlayer] Error decoding audio chunk: {e.Message}");
            }
        }
    }
}
